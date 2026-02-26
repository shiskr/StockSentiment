# This is a sample Python script.
import pandas as pd
from preprocessing.save_file import save_csv
from preprocessing.text_cleaner import clean_text
from training.train_sentiment import train


def main():
    # processed_data = pd.DataFrame(clean_text())
    # save_csv('./data/processed/processed_data.csv', processed_data)
    # data = pd.read_csv('./data/processed/processed_data.csv')
    # print(data['sentiment'].value_counts())
    #
    # label_map = {
    #     "negative": 0,  # SELL
    #     "neutral": 1,  # HOLD
    #     "positive": 2  # BUY
    # }
    #
    # data["label"] = data["sentiment"].map(label_map)
    # print(data.head())
    # data.to_csv('./data/processed/processed_data_labeled.csv', index=False)
    # train()
    from inference.predict import predict
    print(predict("Tesla earnings look weak this quarter"))
    # predict("Tesla earnings look weak this quarter")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
