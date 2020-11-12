% What is the difference between histeq and imadjust functions in Matlab? Play with these functions with various
% input parameters for Camera Man image. Write down your observations in your report and display results.

cameraMan = imread('Camera Man.bmp');
adjusted = imadjust(cameraMan);
adjusted_one = imadjust(cameraMan,[0.3 0.7]);
adjusted_two = imadjust(cameraMan,[0.3 0.7],[0.1 0.9]);
adjusted_three = imadjust(cameraMan,[0.0 0.3],[0 0.6]);
adjusted_four = imadjust(cameraMan,[0.7 1],[0.4 1]);
equalized = histeq(cameraMan);
equalized_one = histeq(cameraMan,128);
equalized_two = histeq(cameraMan,8);
imwrite(adjusted,"a.jpg");
imwrite(adjusted_one,"b.jpg");
imwrite(adjusted_two,"c.jpg");
imwrite(adjusted_three,"c_plus.jpg");
imwrite(adjusted_four,"c_mines.jpg");
imwrite(equalized,"d.jpg");
imwrite(equalized_one,"e.jpg");
imwrite(equalized_two,"f.jpg");

