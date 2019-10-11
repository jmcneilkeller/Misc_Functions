### CHI SQUARED TEST###

chi2 = np.array([factor_level_1,factor_level_1])
chi2_stat, p_val, dof, ex = chi2_contingency(chi2)
print("===Chi2 Stat===")
print(s_chi2_stat)
print("\n")
print("===Degrees of Freedom===")
print(s_dof)
print("\n")
print("===P-Value===")
print(s_p_val)
print("\n")
print("===Contingency Table===")
print(s_ex)

