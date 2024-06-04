

TICKERS=('AAPL' 'ABBV' 'ACN' 'ADBE' 'AEP' 'AFL' 'AIG' 'ALGN'
         'ALL' 'AMAT' 'AMD' 'AMGN' 'AMZN' 'AON' 'APA' 'APD' 'APH'
         'ASML' 'AVB' 'AVGO' 'AXP' 'AZO' 'BA' 'BAC' 'BBY' 'BDX' 'BEN'
         'BIIB' 'BK' 'BKNG' 'BMY' 'BSX' 'BXP' 'C' 'CAG' 'CAH' 'CAT'
         'CB' 'CCL' 'CDNS' 'CE' 'CF' 'CHD' 'CHTR' 'CI' 'CINF' 'CL'
         'CLX' 'CMA' 'CMCSA' 'CME' 'CMG' 'CMI' 'COF' 'COO' 'COP'
         'COST' 'CPB' 'CPRT' 'CRM' 'CSCO' 'CSX' 'CTAS' 'CTSH' 'CVS'
         'CVX' 'D' 'DAL' 'DD' 'DE' 'DFS' 'DG' 'DGX' 'DHI' 'DHR' 'DIS'
         'DLR' 'DLTR' 'DOV' 'DOW' 'DRI' 'DTE' 'DUK' 'DVA' 'DVN' 'DXC'
         'EA' 'EBAY' 'ECL' 'ED' 'EFX' 'EIX' 'EL' 'EMN' 'EMR' 'EOG'
         'EQIX' 'EQR' 'ES' 'ESS' 'ETN' 'ETR' 'EW' 'EXC' 'EXPD' 'EXPE'
         'EXR' 'F' 'FANG' 'FAST' 'FCX' 'FDX' 'FE' 'FFIV' 'FIS' 'FITB'
         'FLS' 'FLT' 'FMC' 'FOX' 'FOXA' 'FRT' 'FTI' 'FTNT' 'FTV' 'GD'
         'GE' 'GILD' 'GIS' 'GL' 'GLW' 'GM' 'GOOG' 'GOOGL' 'GPC' 'GPN'
         'GPS' 'GRMN' 'GS' 'GWW' 'HAL' 'HAS' 'HBAN' 'HBI' 'HCA' 'HCP'
         'HD' 'HES' 'HIG' 'HII' 'HLT' 'HOG' 'HOLX' 'HON' 'HP' 'HPE'
         'HPQ' 'HRB' 'HRL' 'HST' 'HSY' 'HUM' 'IBM' 'ICE' 'IDXX' 'IFF'
         'ILMN' 'INCY' 'INTC' 'INTU' 'IP' 'IPG' 'IPGP' 'IQV' 'IR' 'IRM'
         'ISRG' 'IT' 'ITW' 'IVZ' 'JBHT' 'JCI' 'JEF' 'JKHY' 'JNJ' 'JNPR'
         'JPM' 'JWN' 'K' 'KEY' 'KEYS' 'KHC' 'KIM' 'KLAC' 'KMB' 'KMI'
         'KMX' 'KO' 'KR' 'KSS' 'L' 'LEG' 'LEN' 'LH' 'LHX' 'LIN' 'LKQ'
         'LLY' 'LMT' 'LNC' 'LNT' 'LOW' 'LRCX' 'LUV' 'LW' 'LYB' 'M' 'MA'
         'MAA' 'MAC' 'MAR' 'MAS' 'MCD' 'MCHP' 'MCK' 'MCO' 'MDLZ' 'MDT'
         'MET' 'META' 'MGM' 'MHK' 'MKC' 'MKTX' 'MLM' 'MMC' 'MMM' 'MNST'
         'MO' 'MOS' 'MPC' 'MRK' 'MRO' 'MS' 'MSCI' 'MSFT' 'MSI' 'MTB'
         'MTD' 'MU' 'NCLH' 'NDAQ' 'NEE' 'NEM' 'NFLX' 'NI' 'NKE' 'NKTR'
         'NOC' 'NOV' 'NRG' 'NSC' 'NTAP' 'NTRS' 'NUE' 'NVDA' 'NWL' 'NWS'
         'NWSA' 'O' 'OI' 'OKE' 'OMC' 'ORCL' 'ORLY' 'OXY' 'PAYX' 'PCAR'
         'PEG' 'PEP' 'PFE' 'PFG' 'PG' 'PGR' 'PH' 'PHM' 'PKG' 'PLD' 'PM'
         'PNC' 'PNR' 'PNW' 'PPG' 'PPL' 'PRGO' 'PRU' 'PSA' 'PSX' 'PVH'
         'PWR' 'PXD' 'PYPL' 'QCOM' 'QRVO' 'RCL' 'REG' 'REGN' 'RF' 'RHI'
         'RJF' 'RL' 'RMD' 'ROK' 'ROL' 'ROP' 'ROST' 'RSG' 'RTX' 'SAP'
         'SBAC' 'SBUX' 'SCHW' 'SEE' 'SHOP' 'SHW' 'SJM' 'SLB' 'SLG' 'SNA'
         'SNOW' 'SNPS' 'SO' 'SPG' 'SPGI' 'SQ' 'SRE' 'STI' 'STT' 'STX'
         'STZ' 'SWK' 'SWKS' 'SYF' 'SYK' 'SYY' 'T' 'TAP' 'TDG' 'TEL' 'TFX'
         'TGT' 'TJX' 'TMO' 'TMUS' 'TPR' 'TRIP' 'TROW' 'TRV' 'TSCO' 'TSLA'
         'TSN' 'TTWO' 'TXN' 'TXT' 'UA' 'UAA' 'UAL' 'UDR' 'UHS' 'ULTA'
         'UNH' 'UNM' 'UNP' 'UPS' 'URI' 'USB' 'V' 'VFC' 'VLO' 'VMC' 'VNO'
         'VRSK' 'VRSN' 'VRTX' 'VTR' 'VZ' 'WAB' 'WAT' 'WBA' 'WDC' 'WEC'
         'WELL' 'WFC' 'WHR' 'WM' 'WMB' 'WMT' 'WRK' 'WU' 'WY' 'WYNN'
         'XEL' 'XOM' 'XRAY' 'XRX' 'XYL' 'YUM' 'ZBH' 'ZION' 'ZM' 'ZTS')

#TICKERS=('AAPL' 'ADBE' 'AMAT' 'AMD' 'AMZN' 'ASML' 'AVGO' 'CDNS' 
#         'CRM' 'CSCO' 'CTSH' 'EA' 'EBAY' 'EQIX' 'FFIV' 'FIS' 
#         'FTNT' 'GOOG' 'GOOGL' 'HPQ' 'HPE' 'IBM' 'INTC' 'INTU' 
#         'IPGP' 'ISRG' 'JKHY' 'JNPR' 'KLAC' 'LRCX' 'MA' 'MCHP' 
#         'MSFT' 'MSI' 'MU' 'NTAP' 'NVDA' 'ORCL' 'PAYX' 'PCAR' 
#         'PYPL' 'QCOM' 'QRVO' 'ROK' 'SNPS' 'SNOW' 'SQ' 'SWKS' 
#         'TEL' 'TXN' 'VRSN' 'WDC' 'ZM')

#TICKERS=('META' 'AAPL' 'MSFT' 'AMZN' 'GOOG')
#TICKERS=('AAPL')

for ((i=0; i<100; i++))
do
    python3 transformer.py --tickers "${TICKERS[@]}"
done

#for ((i=7; i<${#TICKERS[@]}; i++))
#do
#    # Pass a subset of TICKERS from index 0 to i
#    python3 transformer.py --tickers "${TICKERS[@]:0:i+1}"
#    python3 transformer.py --tickers "${TICKERS[@]:0:i+1}"
#    python3 transformer.py --tickers "${TICKERS[@]:0:i+1}"
#    
#done
