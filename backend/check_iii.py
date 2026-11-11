from test_parse_appendix import df_appendix

print(df_appendix[df_appendix['subdistrict_block'] == 'III'][['raw_line', 'page_number']])

