# Project 10: Predict Online Gaming Behavior

## Mô tả Project
Project phân tích và dự đoán hành vi chơi game trực tuyến sử dụng Random Forest, nhằm:
- Khám phá các pattern trong hành vi gaming
- Xây dựng model machine learning để dự đoán mức độ engagement của người chơi
- Thực hiện nghiên cứu trong lĩnh vực gaming analytics

## Cấu trúc Project
```
DAV_FinalProject/
├── Dataset/
│   └── online_gaming_behavior_dataset.csv
├── gaming_behavior_analysis.ipynb
├── requirements.txt
├── README.md
└── Requirement.png
```

## Cài đặt

1. Clone hoặc download project
2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Khởi động Jupyter Notebook:
```bash
jupyter notebook
```

4. Mở file `gaming_behavior_analysis.ipynb`

## Nội dung Analysis

### 1. Data Exploration
- Thống kê mô tả dataset
- Phân tích missing values
- Hiểu cấu trúc dữ liệu gaming behavior

### 2. Exploratory Data Analysis (EDA)
- Visualization phân phối engagement levels
- Phân tích correlation giữa các features
- Pattern analysis theo genre, age, gender

### 3. Machine Learning Model
- Preprocessing dữ liệu
- Random Forest Classification
- Hyperparameter tuning với GridSearchCV
- Model evaluation và feature importance

### 4. Gaming Analytics Insights
- Player segmentation analysis
- Business recommendations
- Churn risk analysis
- Strategies cho retention và monetization

### 5. Model Deployment
- Function để predict engagement cho player mới
- Example usage và testing

## Key Features của Dataset

- **PlayerID**: ID người chơi
- **Age**: Tuổi
- **Gender**: Giới tính  
- **Location**: Vị trí địa lý
- **GameGenre**: Thể loại game
- **PlayTimeHours**: Tổng giờ chơi
- **InGamePurchases**: Mua hàng trong game
- **GameDifficulty**: Độ khó game
- **SessionsPerWeek**: Số session/tuần
- **AvgSessionDurationMinutes**: Thời lượng session trung bình
- **PlayerLevel**: Level người chơi
- **AchievementsUnlocked**: Số achievement đã mở
- **EngagementLevel**: Mức độ engagement (Target variable)

## Kết quả chính

1. **Model Performance**: Random Forest đạt độ chính xác cao trong việc dự đoán engagement level
2. **Top Factors**: SessionsPerWeek, PlayTimeHours, PlayerLevel là các yếu tố quan trọng nhất
3. **Business Insights**: Recommendations cụ thể cho retention, game design, marketing

## Ứng dụng thực tế

- **Game Development**: Optimize game design dựa trên engagement patterns
- **Marketing**: Target campaigns theo player segments
- **Player Experience**: Personalize content và difficulty
- **Business Strategy**: Improve retention và monetization
