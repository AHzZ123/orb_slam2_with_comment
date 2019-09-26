/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        // 设置进程间的访问标志 告诉Tracking线程，LocalMapping线程正在处理新的关键帧，处于繁忙状态
        // LocalMapping线程处理的关键帧都是Tracking线程发过的
        // 在LocalMapping线程还没有处理完关键帧之前Tracking线程最好不要发送太快
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        // 等待处理的关键帧
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            // 计算关键帧特征点的词典单词向量，将关键帧插入地图
            ProcessNewKeyFrame();

            // Check recent MapPoints
            // 对新添加的地图点融合对于ProcessNewKeyFrame和CreateNewMapPoints中最近添加的MapPoints进行检查剔除	 
            MapPointCulling();

            // Triangulate new MapPoints
            // 创建新的地图点 相机运动过程中与相邻关键帧通过三角化恢复出一些新的地图点MapPoints	
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                // 相邻帧地图点融合检查并融合当前关键帧与相邻帧（两级相邻）重复的MapPoints
                SearchInNeighbors();
            }

            mbAbortBA = false;

            // 已经处理完队列中的最后的一个关键帧，并且闭环检测没有请求停止LocalMapping
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                // 关键帧融合 检测并剔除当前帧相邻的关键帧中冗余的关键帧 Check redundant local Keyframes
                // 剔除的标准是：该关键帧的90%的MapPoints可以被其它关键帧观测到		
			    // Tracking中先把关键帧交给LocalMapping线程
			    // 并且在Tracking中InsertKeyFrame函数的条件比较松，交给LocalMapping线程的关键帧会比较密
                // 在这里再删除冗余的关键帧
                KeyFrameCulling();
            }

            // 将当前帧加入到闭环检测队列中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        // 等待线程空闲 完成一帧关键帧的插入融合工作
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

/**
 * @brief 处理列表中的关键帧
 * 
 * - 计算Bow，加速三角化新的MapPoints
 * - 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
 * - 插入关键帧，更新Covisibility图和Essential图
 * @see VI-A keyframe insertion
 */
void LocalMapping::ProcessNewKeyFrame()
{
    // 从缓冲队列取出一帧待处理的关键帧
    // tracking线程向localmapping中插入关键帧存在该队列中
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    // 计算关键帧的词袋
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    // 遍历与关键帧匹配的特征点
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    // 跟踪局部地图过程中新匹配上的MapPoints和当前关键帧绑定
    // 在TrackLocalMap函数中将局部地图中的MapPoints与当前帧进行了匹配
    // 但是没有将匹配上的地图点与新建的关键帧进行关联，这里将其关联起来
    // （即被观测帧拥有匹配点的信息，但匹配点没有添加被观测帧的信息）
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                // 对仍未关联当前帧信息的的地图点进行处理（单目观测到的地图点都需要处理）
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    //让mappoint知道自己可以被哪些keyframe看到
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    //更新此mappoint参考帧光心到mappoint平均观测方向以及观测距离范围
                    pMP->UpdateNormalAndDepth();
                    //在此mappoint能被看到的特征点中找出最能代表此mappoint的描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                // （双目创建的地图点会关联到该关键帧）
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    // 当前帧生成的MapPoints
                    // 将双目或RGBD跟踪过程中新插入的MapPoints放入mlpRecentAddedMapPoints，等待MapPointCulling函数检查
                    // CreateNewMapPoints函数中通过三角化也会生成MapPoints
                    // 这些MapPoints都会经过MapPointCulling函数的检验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    // 更新关键帧间的连接关系，Covisibility图和Essential图(tree)
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    // 将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

/**
 * @brief 剔除ProcessNewKeyFrame(不在帧上的地图点 进入待查list)和
 *             CreateNewMapPoints(两帧三角变换产生的新地图点进入 待查list)
 *             函数中引入的质量不好的MapPoints
 * @see VI-B recent map points culling
 */	
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // 看到该地图点的帧数
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // VI-B 条件1：
		// 跟踪到该MapPoint的Frame数相比预计可观测到该MapPoint的Frame数的比例需大于25%
	    // IncreaseFound / IncreaseVisible < 25%，注意不一定是关键帧。
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // VI-B 条件2：
        // 从该点建立开始，到现在已经过了不小于2个关键帧
	    // 但是观测到该点的关键帧数却不超过cnThObs帧，那么该点检验不合格
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 从建立该点开始，已经过了3个关键帧而没有被剔除，则认为是质量高的点
	    // 因此没有SetBadFlag()，仅从队列中删除，放弃继续对该MapPoint的检测
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

/**
 * @brief 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints
 *  根据当前关键帧恢复出一些新的地图点，不包括和当前关键帧匹配的局部地图点（已经在ProcessNewKeyFrame中处理）
 *  先处理新关键帧与局部地图点之间的关系，然后对局部地图点进行检查，
 *  最后再通过新关键帧恢复 新的局部地图点：CreateNewMapPoints()
 * 
 * 步骤1：在当前关键帧的 共视关键帧 中找到 共视程度 最高的前nn帧 相邻帧vpNeighKFs
 * 步骤2：遍历和当前关键帧 相邻的 每一个关键帧vpNeighKFs	
 * 步骤3：判断相机运动的基线在（两针间的相机相对坐标）是不是足够长
 * 步骤4：根据两个关键帧的位姿计算它们之间的基本矩阵 F =  inv(K1 转置) * t12 叉乘 R12 * inv(K2)
 * 步骤5：通过帧间词典向量加速匹配，极线约束限制匹配时的搜索范围，进行特征点匹配	
 * 步骤6：对每对匹配点 2d-2d 通过三角化生成3D点,和 Triangulate函数差不多	
 *  步骤6.1：取出匹配特征点
 *  步骤6.2：利用匹配点反投影得到视差角   用来决定使用三角化恢复(视差角较大) 还是 直接2-d点反投影(视差角较小)
 *  步骤6.3：对于双目，利用双目基线 深度 得到视差角
 *  步骤6.4：视差角较大时 使用 三角化恢复3D点
 *  步骤6.4：对于双目 视差角较小时 二维点 利用深度值 反投影 成 三维点    单目的话直接跳过
 *  步骤6.5：检测生成的3D点是否在相机前方
 *  步骤6.6：计算3D点在当前关键帧下的重投影误差  误差较大跳过
 *  步骤6.7：计算3D点在 邻接关键帧 下的重投影误差 误差较大跳过
 *  步骤6.9：三角化生成3D点成功，构造成地图点 MapPoint
 *  步骤6.9：为该MapPoint添加属性 
 *  步骤6.10：将新产生的点放入检测队列 mlpRecentAddedMapPoints  交给 MapPointCulling() 检查生成的点是否合适
 * @see  
 */	
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    // 双目/深度共视帧数量
    int nn = 10;
    // 单目
    if(mbMonocular)
        nn=20;
    // 找到共视程度最高的nn帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    // 外参矩阵
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    // 得到当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    // 相机内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    // 匹配点查找范围
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    // 遍历和当前关键帧相邻的关键帧
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        // 检查是否存在新的关键帧
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        // 判断基线长度是否足够长
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        // 双目/深度
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        // 单目
        else
        {
            // 景深中值
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            // 基线和景深的比值
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            // 特别远（比例特别小），那么不考虑当前邻接的关键帧，不生成3D点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        // F =  inv(K1 转置)*E*inv(K2) = inv(K1 转置) * t12 叉乘 R12 * inv(K2)
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        // 通过帧间词典向量加速匹配，极线约束限制匹配时的搜索范围，进行特征点匹配		
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        // 相邻关键帧的旋转矩阵和平移向量
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        // 相机内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            // 匹配对下标
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0; 

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            // 两特征点在相机坐标系下的归一化坐标
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            // 转换到同一坐标系
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            // 这里计算归一化平面中的两特征点夹角，即14讲图7-9的夹角
            // 夹角越大，不确定度越小；夹角越小，不确定度越大，所以夹角不宜过小
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            // 余弦值为1时，角度为0，这里初始化为一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            // cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)表明视差角正常
            // cosParallaxRays < cosParallaxStereo表明视差角很小
            // 夹角较大时（cosParallaxRays）用三角法恢复3D点，视差角大时用双目恢复3D点（双目以及深度有效）
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                // p1 = k × [R1 t1] × D       k逆 × p1 =  [R1 t1] × D     x1 = T1 × D    x1叉乘x1 = x1叉乘T1 × D = 0
                // p2 = k × [ R2 t2]  × D     k逆 × p2 =  [R2 t2] × D     x2 = T2 × D    x2叉乘x2 = x2叉乘T2 × D = 0
                //
                //p = ( x,y,1)
                //其叉乘矩阵为
                //  叉乘矩阵 = [0  -1  y;                T0
                //             1   0  -x;           *  T1  *D  ===>( y * T2 - T1 ) *D = 0
                //             -y   x  0 ] 		    T2                  ( x * T2 - T0 ) *D = 0
                //一个点两个方程  两个点 四个方程  A × D =0  求三维点 D  对 A奇异值分解
                // 参考：https://www.cnblogs.com/FangLai-you/p/11276148.html
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            // 对于双目 视差角较小时 二维点 利用深度值 反投影 成 三维点
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            // 检查生成的3d点是否在相机前方
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            // 检查第一帧的重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            // 检查第二帧的重投影误差
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            // 检查尺度一致性
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            // 深度比值和 两幅图像下的金字塔层级比值应该相差不大
		    // |ratioDist/ratioOctave |<ratioFactor
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

/**
 * @brief     检查并融合 当前关键帧 与 相邻帧（一级二级相邻帧）重复的地图点 MapPoints
 * 步骤1：获得当前关键帧在covisibility帧连接图中权重排名前nn的一级邻接关键帧(按观测到当前帧地图点次数选取)
 * 步骤2：获得当前关键帧在 其一级相邻帧 在 covisibility图中权重排名前5 的二级邻接关键帧
 * 步骤3：将当前帧的 地图点MapPoints 分别与 其一级二级相邻帧的 地图点 MapPoints 进行融合(保留观测次数最高的)
 * 步骤4：找到一级二级相邻帧所有的地图点MapPoints 与当前帧 的  地图点MapPoints 进行融合
 * 步骤5：更新当前帧 地图点 MapPoints 的描述子，深度，观测主方向等属性
 * 步骤5：更新当前 与其它帧的连接关系 (  观测到互相的地图点的次数等信息 )
 * @return  无
 */	
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    // 找到当前帧一级相邻和二级相邻的关键帧
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
	// 将当前帧的 地图点MapPoints 分别与 其一级二级相邻帧的 地图点 MapPoints 进行融合
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        // 投影当前帧的MapPoints到相邻关键帧pKFi中，在附加区域搜索匹配关键点，并判断是否有重复的MapPoints
        // 1.如果MapPoint能匹配关键帧的特征点，并且该点有对应的MapPoint，那么将两个MapPoint合并（选择观测数多的）
        // 2.如果MapPoint能匹配关键帧的特征点，并且该点没有对应的MapPoint，那么为该点添加MapPoint
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    // 将一级二级相邻帧所有的地图点MapPoints 与当前帧（的MapPoints）进行融合
    // 遍历每一个一级邻接和二级邻接关键帧 找到所有的地图点
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

// true：tracking正在插入关键帧
// false：tracking插入关键帧已经完成
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    // 如果可以添加关键帧，则Tracking线程添加关键帧时，必须保证LocaMapping不能被终止，此时mbNotStop需要为真。
    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    // 观测帧个数 > 3
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            // 尺度约束，要求MapPoint在该局部关键帧的特征尺度大于（或近似于）其它关键帧的特征尺度
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
