import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
#define an array containing the age
age_array=np.random.randint(25,55,60)
 Define an array containing salesamount values
income_array=np.random.randint(300000,700000,3000000)
fig = go.Figure()
fig.add_trace(go.Scatter(x=age_array, y=income_array, mode='markers', marker=dict(color='blue')))
fig.update_layout(title='sales of every salesman', xaxis_title='amount of sales made', yaxis_title='age of the salesman')
fig.show()

#lineplot of bikes soled per month
#numberofbicyclessold_array=[50,100,40,150,160,70,60,45]
#months_array=["Jan","Feb","Mar","April","May","June","July","August"]
#fig1=go.Figure()
#fig1.add_trace(go.Scatter(x=months_array, y=numberofbicyclessold_array, mode='lines', marker=dict(color='green')))
#fig1.update_layout(title='sales trend',xaxis_title='months',yaxis_title='sales')
##fig1.show()

#bar plot
#score of a student in each grade
#score_array=[80,90,56,88,95]
# Define an array containing Grade names
#grade_array=['Grade 6','Grade 7','Grade 8','Grade 9','Grade 10']
#fig_bar= go.Figure()
#fig_bar.add_trace(go.Bar(x=grade_array, y=score_array, marker=dict(color='blue')))
#fig_bar.show()
#fig_bar = px.bar(x=grade_array, y=score_array, title='score per grade')
#fig_bar.show()

#histogram
#heights_array = np.random.normal(160, 11, 200)
#fig_hist=px.histogram(x=heights_array, title='distribution of heights')
#fig_hist.show()

#bubble plot
#crime_details = {'City' : ['Chicago', 'Chicago', 'Austin', 'Austin','Seattle','Seattle'],'Numberofcrimes' : [1000, 1200, 400, 700,350,1500],'Year' : ['2007', '2008', '2007', '2008','2007','2008'],}
#df=pd.DataFrame(crime_details)
#bub_data = df.groupby('City')['Numberofcrimes'].sum().reset_index()
#fig_bub=px.scatter(bub_data, x="City", y="Numberofcrimes", size="Numberofcrimes",hover_name="City", title='Crime Statistics', size_max=60)
#print(bub_data)
#fig_bub.show()

#pie plot
#exp_percent= [20, 50, 10,8,12]
#house_holdcategories = ['Grocery', 'Rent', 'School Fees','Transport','Savings']
#fig_pie=px.pie(values=exp_percent, names=house_holdcategories, title='household expenditures')
#fig_pie.show()


