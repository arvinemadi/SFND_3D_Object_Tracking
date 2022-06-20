
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "      id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, .5, currColor);
        sprintf(str2, "      xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, .5, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // kptMatches property of the boundingBox structure is augmented with the keypoint matches
    // First iterate over all the input matched keypoints and calculate mean and std of the distance of the matched keypoint between the two successive frame [if within BB ROI]
    // Then iteratre again over all the matched keypoints and pick only the matched keypoints that are within 1 standad deviation of the mean - if guassian it would be 70% of the points
    int n = 0;
    double SS = 0;
    double S = 0;
    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        cv::KeyPoint curr_keypoint = kptsCurr[it->trainIdx];
        cv::KeyPoint prev_keypoint = kptsPrev[it->queryIdx];
        if(boundingBox.roi.contains(curr_keypoint.pt))
        {
            n++;
            double change = cv::norm(curr_keypoint.pt - prev_keypoint.pt);
            
            S  += change;
            SS += change * change;
        }
    }
    double mean_change = S / n;
    double variance_change = SS / n - mean_change * mean_change;
    double std_change = sqrt(variance_change);
    for(auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        cv::KeyPoint curr_keypoint = kptsCurr[it->trainIdx];
        cv::KeyPoint prev_keypoint = kptsPrev[it->queryIdx];
        if(boundingBox.roi.contains(curr_keypoint.pt))
        {
            double change = cv::norm(curr_keypoint.pt - prev_keypoint.pt);
            if (change < mean_change + 1 * std_change)
            {
                boundingBox.kptMatches.push_back(*it);
            }
        }
    }   
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    double sum_ratios = 0;
    double sum_squared_ratios = 0;
    int n = 0;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop
            double minDist = 100.0; // min. required distance
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio =  distCurr / distPrev;
                sum_ratios += distRatio;
                sum_squared_ratios += distRatio * distRatio;
                n++;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts
    double mean_distRatio = sum_ratios / n;
    double variance_distRatio =  sum_squared_ratios / n - mean_distRatio * mean_distRatio;
    double std_distRatio = sqrt(variance_distRatio);
    cout << "Mean of disRatios is " << mean_distRatio << " and with a standard deviation of " << std_distRatio << endl;
    vector<double> filtered_distRatios;
    for(auto& d : distRatios)
    {
        if(fabs(d - mean_distRatio) < 1 * std_distRatio)
            filtered_distRatios.push_back(d);
    }
    // only continue if list of distance ratios is not empty
    if (filtered_distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(filtered_distRatios.begin(), filtered_distRatios.end());
    long medIndex = floor(filtered_distRatios.size() / 2.0);
    double medDistRatio = filtered_distRatios.size() % 2 == 0 ? (filtered_distRatios[medIndex - 1] + filtered_distRatios[medIndex]) / 2.0 : filtered_distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    cout << "Median disRatio is " << medDistRatio << endl;
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // EOF STUDENT TASK
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // simple logic: sort the lidar points O(nlogn) number of lidar points expected to be small if too many points can limit the numbers to be used
    // iterate over the array and if the distance of a single point to the next point was > 0.1 it is most likely an outlier than can be neglected
    // logic is that if there are two points or more it cannot be probably neglected
    sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), [](LidarPoint& a, LidarPoint&b){return a.x < b.x;});
    sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), [](LidarPoint& a, LidarPoint&b){return a.x < b.x;});
    int p_p = 0, p_c = 0;
    while(p_p + 1 < lidarPointsPrev.size() && lidarPointsPrev[p_p + 1].x - lidarPointsPrev[p_p].x > 0.1)    p_p++;
    while(p_c + 1 < lidarPointsCurr.size() && lidarPointsCurr[p_c + 1].x - lidarPointsCurr[p_c].x > 0.1)    p_c++;

    double d0 = lidarPointsPrev[p_p].x;
    double d1 = lidarPointsCurr[p_c].x;
    double dT = 1 / frameRate;
    double velocity = (d0 - d1) / dT;
    TTC = d1 / velocity;
    cout << "d1 is " << d1 << " d0 is << " << d0 << " Velocity is :" << velocity << endl;
    cout << "Lidar Based TTC is :" << TTC << endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Iterate over Iterate keypoints and for each check if it is shared between two bounding boxes of current and previous frame
    // increment index [i][j] of a matrix where i is index of prevoius bounding boxes and j is the index of the current frame bounding box
    // After this is done for all keypoints - iteratre over the matrix and for each bounding box of the previous frame find the index of the box
    // that shared the most keypoints - after finding the index, assing its boxId to the map
    int n_curr_box = (currFrame.boundingBoxes).size();
    int n_prev_box = (prevFrame.boundingBoxes).size();
    vector<vector<int>> n_shared_keypoints (n_prev_box, vector<int> (n_curr_box, 0));
    for(auto it = matches.begin(); it != matches.end(); it++)
    {
        for(int i = 0; i < (prevFrame.boundingBoxes).size(); i++)
        {   
            BoundingBox& prev_box = prevFrame.boundingBoxes[i];
            if (prev_box.roi.contains(prevFrame.keypoints[it->queryIdx].pt))
                for(int j = 0; j < (currFrame.boundingBoxes).size(); j++)
                {
                    BoundingBox& curr_box = currFrame.boundingBoxes[j];
                    if (curr_box.roi.contains(currFrame.keypoints[it->trainIdx].pt))
                        n_shared_keypoints[i][j]++;
                }
        }
    }
    for(int i = 0; i < (prevFrame.boundingBoxes).size(); i++)
    {   
        int max_shared = -1;
        int prev_boxId = prevFrame.boundingBoxes[i].boxID;
        for(int j = 0; j < (currFrame.boundingBoxes).size(); j++)
        {
            if(n_shared_keypoints[i][j] > max_shared)
            {
                    max_shared = n_shared_keypoints[i][j];
                    int curr_boxId = currFrame.boundingBoxes[j].boxID;
                    bbBestMatches[prev_boxId] = curr_boxId;
            }
        }
    }
}
