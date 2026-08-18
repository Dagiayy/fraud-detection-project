

st.sidebar.download_button('Export Alerts CSV', data=df_filtered.to_csv(index=False), file_name='fraud_alerts.csv', mime='text/csv')
