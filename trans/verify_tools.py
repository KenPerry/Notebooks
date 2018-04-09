from trans.data import GetData
gd = GetData()


def verify_df(df, v_df, cols=None, debug=False, **params):
    if "thresh" in params:
        thresh = params["thresh"]
    else:
        thresh = 0

    (min_d, max_d) = (v_df.index.min(), v_df.index.max())
    if debug:
        print("Verified df: ({}, {}), shape {}".format(min_d, max_d, v_df.shape))
        print(df.columns)

        print("Candidate df: : ({}, {}), shape {}".format( df.index.min(), df.index.max(), df.shape))
        print(v_df.columns)

    
    # Output the verified df to a csv for hand-verification
    v_df.to_csv("/tmp/verify.csv")

    if debug:
        verify_df_diagnose(df, v_df, cols, debug, **params)
        
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
   
def verify_df_diagnose(df, v_df, cols=None, debug=False, **params):
    if "thresh" in params:
        thresh = params["thresh"]
    else:
        thresh = 0
        
    (min_d, max_d) = (v_df.index.min(), v_df.index.max())
     
    if (not cols == None):
        diff = abs( df.loc[ min_d:max_d, cols] - v_df.loc[:, cols] )
    else:
        diff = df.loc[ min_d:max_d, v_df.columns] -  v_df.loc[:,:] 

    print("Dates with difference: ", diff.index[ (abs(diff)  > thresh).any(axis=1) ])
    return abs(diff).max(axis=1)
