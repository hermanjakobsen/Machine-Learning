from pandas import read_csv

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv"
names = ['area', 'perimeter', 'compactness', 'lenght_of_kernel','width_of_kernel', 'asymmetry_coefficient', 'lenght_of_kernel_grove', 'class']
dataset = read_csv(url, names=names)

print(dataset)
