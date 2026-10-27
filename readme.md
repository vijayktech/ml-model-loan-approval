Install virtual env in current directory

> python -m venv mysteamlit

Activate mysteamlit
> cd mysteamlit
> activate.bat

Install all dependancy 
> pip install -r requirements.txt

run model file
>py train_model.py

It creates a file loan_decision_tree_model.pkl in root directory.

Run the application
> streamlit run app.py