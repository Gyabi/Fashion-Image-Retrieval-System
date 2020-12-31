import numpy as np
import pandas as pd
color_file_path = "features/"+"DB"+"/"+"DB2"+"_"+"color"+".csv"
type_file_path = "features/"+"DB"+"/"+"DB2"+"_"+"type"+".csv"

color_f = np.loadtxt(color_file_path, delimiter=",",dtype="float32")
type_f = np.loadtxt(type_file_path, delimiter=",",dtype="float32")

black_color_f = color_f[:8]
blue_color_f = color_f[8:16]
white_color_f  = color_f[16:24]
red_color_f = color_f[24:32]

black_type_f,blue_type_f,white_type_f,red_type_f = np.split(type_f[:32], 4)

x = np.array([np.concatenate([a,b]) for (a,b) in zip(color_f, type_f)])
black_x_f,blue_x_f,white_x_f,red_x_f = np.split(x[:32], 4)

output = [[0]*4 for i in range(7)]
for i, x_ in enumerate([black_x_f,blue_x_f,white_x_f,red_x_f]):

    output[0][i] = np.average(np.std(x_,axis=0))
    output[1][i] = np.average(np.std(x_[[0,2,3,4,5]],axis=0))
    output[2][i] = np.average(np.std(x_[[1,2,3,6,7]],axis=0))
    output[3][i] = np.average(np.std(x_[[0,1,3,5,7]],axis=0))
    output[4][i] = np.average(np.std(x_[[0,1,2,4,6]],axis=0))
    output[5][i] = np.average(np.std(x[:32],axis=0))
    output[6][i] = np.average(np.std(x,axis=0))

output = np.array(output)
output = pd.DataFrame(output)
output.columns = ["black", "blue", "white", "red"]
# output.index = ["all", "pi/2_4way", "diagonal_4way","center&back", "right&left", "diagonal_center","diagonal_back", "onry_center","onry_back", "center+rl","onry_back_rl"]
output.index = ["all","not_center", "not_back","not_left","not_right","4colors","dataset"]
print(output)
output.to_csv("standard_deviation.csv")

for i, color in enumerate([black_color_f, blue_color_f, white_color_f, red_color_f]):
    color = color[:4]
    if i == 0:
        a = color
    else:
        a = np.concatenate([a, color])
print(np.average(np.std(a, axis=0)))