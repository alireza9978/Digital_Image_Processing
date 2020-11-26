
image = imread('Lena.bmp');
one = imnoise(image,'gaussian',0,0.01);
two = imnoise(image,'gaussian',0,0.05);
three = imnoise(image,'gaussian',0,0.10);

imwrite(one,"lena_gaussian_01.jpg");
imwrite(two,"lena_gaussian_05.jpg");
imwrite(three,"lena_gaussian_10.jpg");