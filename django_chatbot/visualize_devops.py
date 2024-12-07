import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(chatbot_index, metric_type, y_name, metric_name):
    conn = sqlite3.connect('db.sqlite3')
    query = "select * from chatbot_devopsmetrics where chatbot_index=? and metric_type=?"
    df = pd.read_sql_query(query, conn, params=(chatbot_index, metric_type))

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.strftime('%b %d\n%I:%M:%S %p')

    ax=df.plot(x='created_at', y='metric_value', xlabel="Date", ylabel=y_name, title=f"{metric_name} for {chatbot_index}", legend=False)

    plt.show() 
    conn.close()

plot_metrics("technology-index"       , 'chatbotcounter',"Chat Usage Frequency","Chatbot Usage Frequency Over Time")
plot_metrics("academic-index"         , 'chatbotcounter',"Chat Usage Frequency", "Chatbot Usage Frequency Over Time")
plot_metrics("health-and-safety-index", 'chatbotcounter',"Chat Usage Frequency","Chatbot Usage Frequency Over Time")
plot_metrics("services-index"         , 'chatbotcounter',"Chat Usage Frequency","Chatbot Usage Frequency Over Time")
plot_metrics("general-index"          , 'chatbotcounter',"Chat Usage Frequency", "Chatbot Usage Frequency Over Time")

plot_metrics("technology-index"       , 'avgresponsetime',"Average Response Time (s)", "Average Response Time Over Time")
plot_metrics("academic-index"         , 'avgresponsetime',"Average Response Time (s)", "Average Response Time Over Time")
plot_metrics("health-and-safety-index", 'avgresponsetime',"Average Response Time (s)", "Average Response Time Over Time")
plot_metrics("services-index"         , 'avgresponsetime',"Average Response Time (s)", "Average Response Time Over Time")
plot_metrics("general-index"          , 'avgresponsetime',"Average Response Time (s)", "Average Response Time Over Time")

plot_metrics("technology-index"       , 'avgchatlength',"Average Chat Length (# Q/A Pairs)", "Average Chat Length Over Time")
plot_metrics("academic-index"         , 'avgchatlength',"Average Chat Length (# Q/A Pairs)", "Average Chat Length Over Time")
plot_metrics("health-and-safety-index", 'avgchatlength',"Average Chat Length (# Q/A Pairs)", "Average Chat Length Over Time")
plot_metrics("services-index"         , 'avgchatlength',"Average Chat Length (# Q/A Pairs)", "Average Chat Length Over Time")
plot_metrics("general-index"          , 'avgchatlength',"Average Chat Length (# Q/A Pairs)", "Average Chat Length Over Time")