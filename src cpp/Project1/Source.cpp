// 2018-01-20. 
// Anisotropic image generator
// 2018-01-24
// added double precision by CV_64F. CV_64F = double!!!
// 2018-02-09
// added signal type choise
// 2018-02-10
// added rect and circle H filter
// 2018-03-03
// added filter2D
// 2018-03-06
// added new filter5 - h = rectangle

//#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Function calculates PSD(Power spectrum density) by fft with two flags
// flag = 0 means to return PSD
// flag = 1 means to return log(PSD)
void CalcPSD(Mat & InputImg, Mat & OutputImg, int flag = 0)
{
	Mat planes[2] = { Mat_<float>(InputImg.clone()), Mat::zeros(InputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix
	split(complexI, planes);            // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	float * p;
	p = planes[0].ptr<float>(0);
	p[0] = 0;
	p = planes[1].ptr<float>(0);
	p[0] = 0;

	//	planes[0].at<float>(0) = 0;
	//	planes[1].at<float>(0) = 0;

	// compute the PSD and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	Mat imgPSD;
	magnitude(planes[0], planes[1], imgPSD);				//imgPSD = sqrt(Power spectrum density)
	pow(imgPSD, 2, imgPSD);									//it needs ^2 in order to get PSD
	OutputImg = imgPSD;

	Mat imglogPSD;
	if (flag)
	{
		imglogPSD = imgPSD + Scalar::all(1);						//switch to logarithmic scale
		log(imglogPSD, imglogPSD);								//imglogPSD = log(PSD)
																//Mat imgPhase;
																//phase(planes[0], planes[1], imgPhase);
		OutputImg = imglogPSD;
	}
}

// the function calculates Gauss window and inverse Gauss window
// disp('Filter type: H = exp(-x^2)')
// flag - indicator of direct or inverse Gauss window
// flag = 0 (default) indicates direct Gauss window
// flag = 1 indicates inverse Gauss window
void CalcGaussWindow(Mat & OutputImg, int WindowSize, double sigma1, double sigma2, int flag = 0)
{
	Mat H1(1, WindowSize, CV_64F, Scalar(0));
	Mat H2(WindowSize, 1, CV_64F, Scalar(0));

	Mat H1_arg(1, WindowSize, CV_64F, Scalar(0));
	Mat H2_arg(WindowSize, 1, CV_64F, Scalar(0));

	double k = -0.5;
	if (flag)
		k = 0.5;
	//float * p = H2_arg.ptr<float>(0);
	double * p1 = H1_arg.ptr<double>(0);
	double * p2 = H2_arg.ptr<double>(0);
	double dx = 2.0*CV_PI / WindowSize;
	double x = -CV_PI;
	for (int i = 0; i < WindowSize; i++)
	{
		//p1[i] = -0.5*(x / sigma1)*(x / sigma1);
		//p2[i] = -0.5*(x / sigma2)*(x / sigma2);
		p1[i] = k*(x / sigma1)*(x / sigma1);
		p2[i] = k*(x / sigma2)*(x / sigma2);
		x += dx;
	}
	exp(H1_arg, H1);
	exp(H2_arg, H2);
	Mat H = H2 * H1;
	OutputImg = H.clone();
}


// the function calculates MarkowProcess Window  and inverse MarkowProcess Window
// disp('Filter type: H = exp(-abs(x))')
// flag - indicator of direct or inverse 
// flag = 0 (default) indicates direct 
// flag = 1 indicates inverse 
void CalcMarkowWindow(Mat & OutputImg, int WindowSize, double sigma1, double sigma2, int flag = 0)
{
	Mat H1(1, WindowSize, CV_64F, Scalar(0));
	Mat H2(WindowSize, 1, CV_64F, Scalar(0));

	Mat H1_arg(1, WindowSize, CV_64F, Scalar(0));
	Mat H2_arg(WindowSize, 1, CV_64F, Scalar(0));

	double k = -0.5;
	if (flag)
		k = 0.5;
	//float * p = H2_arg.ptr<float>(0);
	double * p1 = H1_arg.ptr<double>(0);
	double * p2 = H2_arg.ptr<double>(0);
	double dx = 2.0*CV_PI / WindowSize;
	double x = -CV_PI;
	for (int i = 0; i < WindowSize; i++)
	{
		//p1[i] = -0.5*(x / sigma1)*(x / sigma1);
		//p2[i] = -0.5*(x / sigma2)*(x / sigma2);
		p1[i] = k*abs(x / sigma1);
		p2[i] = k*abs(x / sigma2);
		x += dx;
	}
	exp(H1_arg, H1);
	exp(H2_arg, H2);
	Mat H = H2 * H1;
	OutputImg = H.clone();
}

// the function calculates Rectangle Window
void CalcRectWindow(Mat & OutputImg, int WindowSize, Size RectSize)
{
	RectSize.width = (RectSize.width & -2) + 1;
	RectSize.height = (RectSize.height & -2) + 1;
	Rect roi(WindowSize/2- RectSize.width/2, WindowSize/2- RectSize.height/2, RectSize.width, RectSize.height);

	Mat H(WindowSize, WindowSize, CV_64F, Scalar(0));
	H(roi) = 1;
	int N = countNonZero(H);
	H = H / N;
	OutputImg = H.clone();
}

// the function calculates ellipse Window
void CalcEllipseWindow(Mat & OutputImg, int WindowSize, Size RectSize, bool bCircle = true)
{
	//RectSize.width = (RectSize.width & -2) + 1;
	//RectSize.height = (RectSize.height & -2) + 1;
	//Rect roi(WindowSize / 2 - RectSize.width / 2, WindowSize / 2 - RectSize.height / 2, RectSize.width, RectSize.height);

	Mat H(WindowSize, WindowSize, CV_64F, Scalar(0));
	if(bCircle)
		circle(H, Point(WindowSize / 2, WindowSize / 2), RectSize.width, 255, -1,4);
	else
		ellipse(H, Point(WindowSize / 2, WindowSize / 2), RectSize, 0, 0, 360, 255, -1,4);

	int N = countNonZero(H);
	H = H / N;

	OutputImg = H.clone();
}

// Functions rearranges quadrants of Fourier image  so that the origin is at the image center
void fftshift(const Mat & InputImg, Mat & OutputImg)
{
	// crop the spectrum, if it has an odd number of rows or columns
	OutputImg = InputImg(Rect(0, 0, InputImg.cols & -2, InputImg.rows & -2)).clone();
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = OutputImg.cols / 2;
	int cy = OutputImg.rows / 2;
	Mat q0(OutputImg, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(OutputImg, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(OutputImg, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(OutputImg, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

// filter2DFreq filters an image by H transfer function of filter
void filter2DFreq(const Mat & InputImg, Mat & OutputImg, const Mat & H)
{
	Mat planes[2] = { Mat_<double>(InputImg.clone()), Mat::zeros(InputImg.size(), CV_64F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix

	Mat planesH[2] = { Mat_<double>(H.clone()), Mat::zeros(H.size(), CV_64F) };
	Mat complexH;
	merge(planesH, 2, complexH);         // Add to the expanded another plane with zeros
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);            // this way the result may fit in the source matrix
	split(complexIH, planes);
	OutputImg = planes[0].clone();

	Mat abs0 = abs(planes[0]);
	Mat abs1 = abs(planes[1]);
	
	double MaxReal, MinReal;
	minMaxLoc(abs0, &MinReal, &MaxReal, NULL, NULL);

	double MaxIm, MinIm;
	minMaxLoc(abs1, &MinIm, &MaxIm, NULL, NULL);
	cout << "MaxReal = " << MaxReal << "; MinReal = " << MinReal << endl;
	cout << "MaxIm = " << MaxIm << "; MinIm = " << MinIm << endl;
	cout << endl;
}

int main()
{
	//creating a Random Number Generator object (RNG)
//	RNG rng(0xFFFFFFFF);

	const int WindowSize = 256;

	namedWindow("imgOriginal", WINDOW_AUTOSIZE);
	namedWindow("imgPSD", WINDOW_AUTOSIZE);
	namedWindow("H", WINDOW_AUTOSIZE);
	namedWindow("h", WINDOW_AUTOSIZE);
	namedWindow("imgOut", WINDOW_AUTOSIZE);
	namedWindow("imgOutRoi", WINDOW_AUTOSIZE);
	namedWindow("control");

	//Create track bar for signal type
	int SignalType = 1;
	createTrackbar("SignalType", "control", &SignalType, 1);

	//Create track bar for signal type
	int FilterType = 3;
	createTrackbar("FilterType", "control", &FilterType, 7);

	//Create track bar for sigma
	int sigma1 = 4;
	createTrackbar("0.1*sigma1", "control", &sigma1, 30);
	cvSetTrackbarMin("0.1*sigma1", "control", 1);
	int sigma2 = 4;
	createTrackbar("0.1*sigma2", "control", &sigma2, 30);
	cvSetTrackbarMin("0.1*sigma2", "control", 1);

	//Create track bar for borderSize
	int borderSize = 0;
	createTrackbar("borderSize", "control", &borderSize, 100);



	while (true)
	{
		Mat imgOriginal(WindowSize, WindowSize, CV_8UC1, Scalar(0));
		//	rng.fill(image, RNG::NORMAL, 0, 0.5);
		if (SignalType)
			imgOriginal.at<char>(WindowSize / 2, WindowSize / 2) = 1;
		else
			randn(imgOriginal, 0, 10);

		Mat img;
		copyMakeBorder(imgOriginal, img, borderSize, borderSize, borderSize, borderSize, BORDER_CONSTANT);

		Mat imgPSD;
		CalcPSD(img, imgPSD);
		sqrt(imgPSD, imgPSD);
		sqrt(imgPSD, imgPSD);
		fftshift(imgPSD, imgPSD);

		Mat H, H_shift;
		Mat imgOut;

		switch (FilterType)
		{
		case 0:
			CalcGaussWindow(H, WindowSize + 2 * borderSize, double(sigma1) / 10.0, double(sigma2) / 10.0);
			fftshift(H, H_shift);
			filter2DFreq(img, imgOut, H_shift);
			cout << "Gauss window. H = exp(-x^2)" << endl;
			normalize(H, H, 0, 1, NORM_MINMAX);
			imshow("H", H);
			break;
		case 1:
			CalcMarkowWindow(H, WindowSize + 2 * borderSize, double(sigma1) / 10.0, double(sigma2) / 10.0);
			fftshift(H, H_shift);
			filter2DFreq(img, imgOut, H_shift);
			cout << "Markov window. H = exp(-abs(x))" << endl;
			normalize(H, H, 0, 1, NORM_MINMAX);
			imshow("H", H);
			break;
		case 2:
			CalcRectWindow(H, WindowSize + 2 * borderSize, Size(sigma1, sigma2));
			fftshift(H, H_shift);
			filter2DFreq(img, imgOut, H_shift);
			cout << "Rectangle window. H = rectangle (w = " << sigma1 << ", h = " << sigma2 << ")" << endl;
			normalize(H, H, 0, 1, NORM_MINMAX);
			imshow("H", H);
			break;
		case 3:
		{
			Mat h;
			CalcRectWindow(h, WindowSize + 2 * borderSize, Size(sigma1, sigma2));
			//h.convertTo(h, CV_64F);
			//int N = countNonZero(h);
			//h = h / N;
			img.convertTo(img, CV_64F);
			filter2D(img, imgOut, img.depth(), h);
			img.convertTo(img, CV_8U);
			cout << "Rectangle window. h = rectangle (w = " << sigma1 << ", h = " << sigma2 << ")" << endl;
			normalize(h, h, 0, 1, NORM_MINMAX);
			imshow("h", h);
			break;
		}
		case 4:
			CalcEllipseWindow(H, WindowSize + 2 * borderSize, Size(sigma1, sigma2));
			fftshift(H, H_shift);
			filter2DFreq(img, imgOut, H_shift);
			cout << "Circle window. H = circle (R = " << sigma1 << ")" <<  endl;
			normalize(H, H, 0, 1, NORM_MINMAX);
			imshow("H", H);
			break;
		case 5:
			CalcEllipseWindow(H, WindowSize + 2 * borderSize, Size(sigma1, sigma2), false);
			fftshift(H, H_shift);
			filter2DFreq(img, imgOut, H_shift);
			cout << "Ellipse window. H = ellipse (R1 = " << sigma1 << ", R2 = " << sigma2 << ")" << endl;
			normalize(H, H, 0, 1, NORM_MINMAX);
			imshow("H", H);
			break;
		case 6:
		{
			Mat h;
			CalcEllipseWindow(h, WindowSize + 2 * borderSize, Size(sigma1, sigma2));
			//h.convertTo(h, CV_64F);
			//int N = countNonZero(h);
			//h = h / N;
			img.convertTo(img, CV_64F);
			filter2D(img, imgOut, img.depth(), h);
			img.convertTo(img, CV_8U);
			cout << "Circle window. h = circle (R = " << sigma1 << ")" << endl;
			normalize(h, h, 0, 1, NORM_MINMAX);
			imshow("h", h);
			break;
		}
		case 7:
		{
			Mat h;
			CalcEllipseWindow(h, WindowSize + 2 * borderSize, Size(sigma1, sigma2), false);
			//int N = countNonZero(h);
			//h = h / N;
			img.convertTo(img, CV_64F);
			filter2D(img, imgOut, img.depth(), h);
			img.convertTo(img, CV_8U);
			cout << "Ellipse window. h = ellipse (R1 = " << sigma1 << ", R2 = " << sigma2 << ")" << endl;
			normalize(h, h, 0, 1, NORM_MINMAX);
			imshow("h", h);
			break;
		}
		}

		//int offset = 20;
		int offset = 0;
		Mat imgOutRoi = imgOut(Rect(borderSize+ offset, borderSize + offset, WindowSize- 2*offset, WindowSize- 2*offset)).clone();
		//Mat imgOutRoi = imgOut(Rect(borderSize, borderSize+, WindowSize, WindowSize)).clone();
		//Mat imgOutRoi = imgOut(Rect(2*borderSize, 2*borderSize, WindowSize-2*borderSize, WindowSize-2*borderSize)).clone();

		normalize(img, img, 0, 255, NORM_MINMAX);
		normalize(imgPSD, imgPSD, 0, 1, NORM_MINMAX);
		normalize(imgOut, imgOut, 0, 1, NORM_MINMAX);
		normalize(imgOutRoi, imgOutRoi, 0, 1, NORM_MINMAX);

		imshow("imgOriginal", img);
		imshow("imgPSD", imgPSD);
		imshow("imgOut", imgOut);
		imshow("imgOutRoi", imgOutRoi);

		// Wait until user press some key for 50ms
		int iKey = waitKey(2000);
		//if user press 'ESC' key
		if (iKey == 27)
		{
			break;
		}
	}

	//waitKey(0);
	return 0;
}