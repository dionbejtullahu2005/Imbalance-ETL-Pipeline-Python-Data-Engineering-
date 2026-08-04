import pandas as pd

def get_gj_value(df_gj, nr):
    value = (
        df_gj.loc[df_gj['Nr. rendor'] == nr,
                  "Sasia"].values[0]
    )
    return float(value)

def calculate_gj_summary(df, df_gj):
    gj = {}

    #Energjia e matur
    gj[1] = df["emi"].sum()

    #Energjia e kontraktuar
    gj[2] = df["eknk"].sum()

    #imbalancet
    gj[3] = df.loc[df["ejb"] > 0, "ejb"].sum()
    gj[4] = df.loc[df["ejb"] < 0, "ejb"].sum()

    #pagesat
    gj[5] = df.loc[df["payment"] > 0, "payment"].sum()
    gj[6] = df.loc[df["payment"] < 0, "payment"].sum()

    #energjia hyrese
    gj[7] = get_gj_value(df_gj,7)

    #energjia dalese
    gj[8] = get_gj_value(df_gj,8)

    #Energjia ne llogarine balancuese
    if gj[7] is not None and gj[8] is not None:
        gj[9] = gj[7] - gj[8]
    else:
        gj[9] = None

    #llogaria balancuese
    gj[10] = get_gj_value(df_gj,10)

    #Cmimi balancues
    if gj[9]:
        gj[11]= gj[10] / gj[9]
    else:
        gj[11] = None

    #rialokimi
    if gj[11]:
        gj[12] = abs(gj[11] * gj[1])
    else:
        gj[12] = None

    #pagesa finale
    if gj[12]:
        gj[13] = gj[5] + gj[6] + gj[12]
    else: 
        gj[13] = None

    gj[14] = 0.200
    gj[15] = 4.279
    gj[16] = 0.022

    #cmimi referent
    gj[17] = 88.58

    #pagesat tarifore
    gj[18] = gj[14] * gj[1]
    gj[19] = gj[15] * gj[1]
    gj[20] = gj[16] * gj[1]

    #Nominimet BRE
    gj[21] = get_gj_value(df_gj,21)

    #Pagesa nominimeve
    if gj[21]:
        gj[22] = -(gj[17] * gj[21])
    else:
        gj[22] = None


    return pd.DataFrame(
        {
            "Nr": list(gj.keys()),
            "Python_Value": list(gj.values())
        }
    )
