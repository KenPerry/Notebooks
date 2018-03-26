from trans.data import GetData
gd = GetData()


def verify_df(df, v_df, cols=None, debug=False, **params):
    (min_d, max_d) = (v_df.index.min(), v_df.index.max())
    if debug:
        print("Verified df: ({}, {}), shape {}".format(min_d, max_d, v_df.shape))
        print(df.columns)

        print("Candidate df: : ({}, {}), shape {}".format( df.index.min(), df.index.max(), df.shape))
        print(v_df.columns)

    
    # Output the verified df to a csv for hand-verification
    v_df.to_csv("/tmp/verify.csv")
    
    if (not cols == None):
        return df.loc[ min_d:max_d, cols].equals( v_df.loc[:, cols])
    else:
        return df.loc[ min_d:max_d, v_df.columns].equals( v_df.loc[:,:])

    
def verify_file(df, verified_df_file, cols=None, debug=False,**params):
    """
    Compare DataFrame to one that is stored in a file
    
    Parameters:
    --------------
    df: DataFrame
    verified_df_file: string. Name of pkl file containing verified DataFrame
    
    Returns
    --------
    Boolean
    """                   
    v_df = gd.load_data(verified_df_file)
    return verify_df(df, v_df, cols=cols, debug=debug, **params)
   
