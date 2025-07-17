Requirement: Python 3.12.x
** Window **
>> cd Chatbot
>> python -m venv venv
>> venv\Scripts\activate
>> pip install -r requirements.txt

# For training BiLSTM (predict intent)
>> cd scr
>> cd intent
>> python train.py

After training, file will be saved in scr\intent\model 
>> mv scr/intent/model/* model/

# For training BILSTM slott-filling (take order)
get back to scr or >> cd ..
>> cd order
>> python train.py

After training, file will be saved in scr\order\model 
>> mv scr/order/model/* model/

# Run chatbot demo
>> cd ..
>> python router.py
