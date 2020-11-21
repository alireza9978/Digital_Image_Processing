
image = imread('Elaine.jpg');
one = imnoise(image,'gaussian',0,0.01);
two = imnoise(image,'gaussian',0,0.05);
three = imnoise(image,'gaussian',0,0.10);

imwrite(one,"Elaine_gaussian_01.jpg");
imwrite(two,"Elaine_gaussian_05.jpg");
imwrite(three,"Elaine_gaussian_10.jpg");