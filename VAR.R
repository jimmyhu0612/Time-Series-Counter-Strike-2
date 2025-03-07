library(readxl)
library(vars)

# 設定檔案路徑
file_path <- "C:/Users/jimmyhu/Desktop/研究所_行銷/112-2/時間序列分析/final/onlyCSGOdata_utf8.xlsx"

# 讀取XLSX檔案
data <- read_excel(file_path)

# 檢查資料是否正確讀取
head(data)

# 確保時間序列資料按照日期排序
data <- data[order(data$time), ]

# 建立時間序列物件，假設日期欄位名稱為"date"
start_year <- 2012
start_month <- 9

time_series_data <- ts(data[, c("avg_players", "pctg_gain", "peak_players", "skin_price", "sold_units")], 
                       start = c(start_year, start_month), frequency = 12)

# 檢查時間序列資料
plot(time_series_data)

# 建立VAR模型，選擇適當的滯後期數
var_model <- VAR(time_series_data, p = 1) # 這裡的滯後期數(p)可以根據需要調整

# 顯示VAR模型結果
summary(var_model)
#===============================
library(readxl)
library(vars)

# 設定檔案路徑
file_path <- "C:/Users/jimmyhu/Desktop/研究所_行銷/112-2/時間序列分析/final/onlyCSGOdata_utf8.xlsx"

# 讀取XLSX檔案
data <- read_excel(file_path)

# 檢查資料是否正確讀取
head(data)

# 確保時間序列資料按照日期排序
data <- data[order(data$time), ]

# 建立時間序列物件，假設日期欄位名稱為"date"
start_year <- 2012
start_month <- 9

time_series_data <- ts(data[, c("avg_players", "pctg_gain", "peak_players", "skin_price", "sold_units")], 
                       start = c(start_year, start_month), frequency = 12)

# 檢查時間序列資料
plot(time_series_data)

# 使用信息準則選擇最佳滯後期數
lag_selection <- VARselect(time_series_data, lag.max = 12, type = "const") # lag.max設定為12，可以根據需要調整

# 顯示信息準則的結果
print(lag_selection$selection)

# 根據選擇的最佳滯後期數建立VAR模型
best_lag <- lag_selection$selection["AIC(n)"] # 使用AIC標準選擇的最佳滯後期數
var_model <- VAR(time_series_data, p = best_lag)

# 顯示VAR模型結果
summary(var_model)
