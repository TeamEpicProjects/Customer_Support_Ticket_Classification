import pandas as pd
# sentence feature extracter
from embedding_as_service.text.encode import Encoder
# light model used
import lightgbm as lgb
url = 'https://raw.githubusercontent.com/TeamEpicProjects/Customer_Support_Ticket_Classification/Day_04/ticket_train.csv'
train = pd.read_csv(url)
glove_en = Encoder(embedding='glove', model='crawl_42B_300')
# encoding training dataset
glove_train_vecs = glove_en.encode(texts = list(train['info'].values), pooling='reduce_mean')
final_model = lgb.LGBMClassifier(is_unbalance = True,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     n_estimators = 50)
final_model.fit(glove_train_vecs, train['ticket_type'])