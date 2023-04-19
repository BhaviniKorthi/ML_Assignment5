# ML_Assignment5

## DataFrame 

The table showing the results of all the four models - 
1. VGG (1 block)
2. VGG (3 blocks)
3. VGG (3 blocks) with data augmentation
4. Transfer learning using VGG16 or VGG19

![image](https://user-images.githubusercontent.com/76489649/233083072-09bdc9a4-90fc-48a3-a33d-ff9b0278837a.png)

## Tensorboard  Visualization

### Scalars:

1. VGG (1 block)
2. VGG (3 blocks)
3. VGG (3 blocks) with data augmentation
4. Transfer learning using VGG16 or VGG19

# ML_Assignment5


![image](https://user-images.githubusercontent.com/102377549/233099630-2dff38f8-4f68-4321-add7-e7fb33ef5158.png)

![image](https://user-images.githubusercontent.com/102377549/233099777-83613509-ff7f-4b49-990e-b08f0691b005.png)


### Images:

1. VGG (1 block)
![image](https://user-images.githubusercontent.com/76489649/233095818-42fbc0cd-b874-4321-8c2a-120c053bba70.png)

2. VGG (3 blocks)
![image](https://user-images.githubusercontent.com/76489649/233095882-c80344e4-2c67-4422-b7b9-c1824368af8d.png)


3. VGG (3 blocks) with data augmentation
![image](https://user-images.githubusercontent.com/76489649/233095645-fdc4e5d8-d855-4a8d-b0e8-f8aa90c2eb3d.png)

4. Transfer learning using VGG16 or VGG19


## Questions:

### Are the results as expected? Why or why not?
Yes, the results are as expected. Both the test and train accuracies increases in the order VGG (1 block) < VGG (3 blocks) < VGG (3 blocks) with data augmentation < Transfer learning using VGG16 or VGG19


### Does data augmentation help? Why or why not?
Yes, the data augmentation is helpful because the accuracy of the model increases by using it. Since we are using a dataset with a limited data, the data augmentation plays a crucial in improving the accuracy by introducing some variability into the data.


### Does it matter how many epochs you fine tune the model? Why or why not?
Yes, the number of epochs that we use to tune a model have a significant impact on the performance. Suppose, if we use a higher number of epochs, then it may lead to overfit to the training data. In contrast, if we use lower number of epochs, then model may not learn the data well and lead to underfitting of the data.


### Are there any particular images that the model is confused about? Why or why not?

![sheep5](https://user-images.githubusercontent.com/76489649/233100789-8fcbdf7e-eb55-40af-aa86-31a7d9cf918b.png)
