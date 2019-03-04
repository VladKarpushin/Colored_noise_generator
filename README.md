colored noise generator
==========================

Goal
----

In this tutorial you will learn:

-   how to generate a colored noise

Theory
------
Input signal: 
white noise or delta function.

Filter:
H = GaussWindow, MarkowWindow, RectWindow, EllipseWindow.

h = RectWindow, EllipseWindow.

Result
------
So, totally, we can generate 12 different images (2 * 4 + 2 * 2).
Some results you can see below.

The figure below shows an filter output result when H = gauss, input signal = white noise.

![1](/www/images/H=gauss1_signal=noise.jpg)

The figure below shows an filter output result when H = gauss, input signal = white noise.

![1](/www/images/H=gauss2_signal=noise.jpg)

The figure below shows an filter output result when h = circle, input signal = white noise.

![1](/www/images/h=circle_signal=noise.jpg)

The figure below shows an filter output result when h = square, input signal = white noise.

![2](/www/images/h=square_signal=noise.jpg)

The figure below shows an filter output result when H = circle, input signal = white noise.

![3](/www/images/HH=circle_signal=noise.jpg)

The figure below shows an filter output result when H = circle, input signal = delta function.

![4](/www/images/H=circle_signal=delta.jpg)