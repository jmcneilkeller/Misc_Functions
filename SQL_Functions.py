def sql (query,cursor = c):
    """Runs a SQL query and returns results as a Dataframe. cursor Variable is set to 'c' by default. Make sure your run sql_connect before using or set cursor manually """
    c.execute(query)
    c.fetchall
    df = pd.DataFrame(c.fetchall())
    df.columns = [x[0] for x in c.description]
    return df
