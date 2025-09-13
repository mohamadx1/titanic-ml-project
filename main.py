from src.data_preprocessing import load_data, fill_missing_values
from src.feature_engineering import encode_features, create_features
from src.models import train_model

if __name__ == "__main__":
    # تحميل البيانات
    train, test = load_data("data/train.csv", "data/test.csv")

    # تنظيف
    train = fill_missing_values(train)
    test = fill_missing_values(test)

    # تحويل وإنشاء Features
    train = encode_features(create_features(train))
    test = encode_features(create_features(test))

    # فصل المتغيرات
    X = train.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
    y = train["Survived"]

    # تدريب النموذج
    model, acc = train_model(X, y)
    print(f"Validation Accuracy: {acc:.2f}")
