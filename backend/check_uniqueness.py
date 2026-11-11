from test_parse_appendix import df_appendix
import pandas as pd

print(f"Total rows in appendix: {len(df_appendix)}")
print(f"Unique villages: {df_appendix['name'].nunique()}")
print(f"Unique subdistricts (blocks/ranges): {df_appendix['subdistrict_block'].nunique()}")
print("Subdistricts:", sorted(df_appendix['subdistrict_block'].unique()))

