colored noise generator
==========================

Goal
----

In this tutorial you will learn:

-   how to generate colored noise

Theory
------
Input signal: 
white noise or delta function

Filter:
H = GaussWindow, MarkowWindow, MarkowWindow, RectWindow, EllipseWindow
h = RectWindow, EllipseWindow, 

Result
------
So, totally, we can generate 14 different images (2 * 5 + 2 * 2)
Some result you can see below.

The figure below shows an filter output result when h = circle, input signal = white noise
![Image corrupted by periodic noise](/www/images/h=circle.jpg)

The figure below shows an filter output result when h = square, input signal = white noise
![Power spectrum density showing periodic noise](/www/images/h=square.jpg)

The figure below shows an filter output result when H = circle, input signal = white noise
![Power spectrum density showing periodic noise](/www/images/HH=circle.jpg)
