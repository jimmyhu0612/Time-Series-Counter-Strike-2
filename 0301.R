fert = c(1, 1, 1, 2, 2, 3, 3) #c:建立向量
height = c(74, 68, 77, 76, 80, 87, 91)
initial = c(3, 4, 5, 2, 4, 3, 7)
Data = data.frame(fert, height, initial) #建立dataframe
apply(Data, 2, min) #apply():建立一個function
apply(Data, 2, max) 
apply(Data, 2, mean)
apply(Data, 2, sd)
apply(Data, 2, quantile, prob = c(0.25, 0.75), na.rm = T) #若只要知道一個四分位數就只要打0.25，不用c()
                                              #是否需補補缺失值
apply(Data,2,max)-apply(Data,2,min)
library(e1071)
apply(Data,2,skewness)
apply(Data,2,kurtosis)
