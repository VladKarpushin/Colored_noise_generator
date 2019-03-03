% 2018-01-14
% Anisotropic image generator_matlab
% 2018-01-20
% debluring by inverse filter
% 2018-01-26
% added different filter H
% added different input signals
% 2018-01-28
% Improved H filter design, now max(imag(ifft(H))) = e-17

%close all force
close all hidden, clc, clear all;

%imgA = rgb2gray(imread('input\test2.jpg'));     %thr = 0.4;
%imgA = rgb2gray(imread('input\input.tif'));     %thr = 0.01;


%********************
% image generation  *
%********************

N=512;             % even 2/4/6
Signal_type = 1;

switch Signal_type
    case 0      % white noise
        disp('Signal type: white noise')
        imgA = randn(N,N);
    case 1      % delta function
        disp('Signal type: delta function')
        imgA = zeros(N,N);
        imgA(N/2,N/2) = 1;
end

imgA_fft = fft2(imgA);

imgA_PSD = imgA_fft.*conj(imgA_fft);
imgA_PSD(1,1) = 0;
%imgA_PSD = fftshift(imgA_PSD);

figure, 
subplot(2,2,1);
imshow(imgA, []);
title('img');
subplot(2,2,2);
imshow(imgA_PSD, []);
title('PSD');


%***********
% bluring  *
%***********

nFilter = 1;        % filter ID
sigma1 = 0.6;
sigma2 = 0.6;

nfft = N;
x = -pi:2*pi/nfft:pi-pi/nfft;

switch nFilter
    case 0      % Gauss process
        disp('Filter type: H = exp(-x^2)')
        H1 = exp(-1/2*(x/sigma1).^2)';
        H2 = exp(-1/2*(x/sigma2).^2)';
        H = H1*H2';
    case 1      % Markov process
        disp('Filter type: H = exp(-abs(x))')
        H1 = exp(-1/2*abs(x/sigma1))';      
        H2 = exp(-1/2*abs(x/sigma2))';
        H = H1*H2';
    case 2      % rectangle
        disp('Filter type: H = rectangle')
        w = 11;  %odd 3/5/7
        h = 11;  %odd 
        H1 = zeros(N,1);
        H2 = zeros(N,1);
        H1(N/2-w/2+1:N/2+w/2) = 1;
        H2(N/2-h/2+1:N/2+h/2) = 1;
        H = H1*H2';
    case 3
        disp('Filter type: H = ellipse')
        R = 11;  % odd or even, any
        H = MyCircle(N, R);
end

%figure, plot(x, H1);
%figure, plot(x, H2);
subplot(2,2,3);
imshow(H,[]);
title('filter H');

imgB = ifftshift(H);
imgC = ifft2(imgA_fft .* imgB);
%fprintf('mmax(real = %f)\n',max(max(real(imgC))));
disp(max(max(abs(real(imgC)))));
%disp(min(min(real(imgC))));
disp(max(max(abs(imag(imgC)))));
%disp(min(min(imag(imgC))));
imgC = real(imgC);

subplot(2,2,4);
imshow(imgC,[]);
title('Filtered image');

return;

%************
% debluring *
%************

% it needs to cut endges for better debluring by inverse filter

sigma1 = 0.6;
sigma2 = 0.6;
H1_inv = exp(1/2*(x/sigma1).^2)';
H2_inv = exp(1/2*(x/sigma2).^2)';
%H1_inv = 1./H1;
%H2_inv = 1./H2;
H_inv = H1_inv*H2_inv';
imgD = ifftshift(H_inv);
figure, imshow(H_inv,[]);
title('H inv');

% figure, plot(x, H1_inv);
% title('inverse filter H');

imgE = ifft2(fft2(imgC) .* imgD);
disp(max(max(abs(real(imgE)))));
%disp(min(min(real(imgC))));
disp(max(max(abs(imag(imgE)))));

figure, imshow(abs(real(imgE)),[]);
title('de Blured image');