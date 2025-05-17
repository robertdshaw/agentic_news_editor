# Reset Python paths
$env:PATH = "C:\Python313;C:\Python313\Scripts;" + $env:PATH
# Run the app
python -m streamlit run app.py
