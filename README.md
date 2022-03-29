# DSAI-HW1
2022 DSAI HW1

## Usage
```sh
$ python app.py --training training_data.csv --output submission.csv
```

## Idea

### Data Analysis
- 5-10月用電最高
- 週間用電大於週末 → 形成一個週期
- 110 5月中以後備轉容量波動原因不明，一直持續到近年
- 如果系統運轉淨尖峰能力不變，則備轉容量的波動應該與瞬時尖峰負載波動互補
- [今日用電差不多等於昨日用電](https://www.taipower.com.tw/tc/page.aspx?mid=206&cid=402&cchk=8c59a5ca-9174-4d2e-93e4-0454b906018d)
- 台電未來一週電力供需預測，係參照中央氣象局每日11時所公布**未來一週氣象預報資料**，並依當時**發電機組排程計劃**進行評估作業。
- 台電未來二個月電力供需預測，係依據本公司**負載預測**及**發電機組狀況**，並參考**中央氣象局長期天氣預測資料**修正產生。

### Data Preprocessing
- 台電有釋出一個`data/reserve.csv`，我把他爬下來然後塞到訓練資料，但
過濾掉未來資料

### Feature selection
過去備轉容量

### Model training
先拿一個 baseline model 跑跑看，這裡使用的是 [Ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
