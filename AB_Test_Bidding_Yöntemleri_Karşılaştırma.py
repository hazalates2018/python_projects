import pandas as pd
df1 = pd.read_excel("C:\\Users\\hazal\\Downloads\\AB_TEST.xlsx",sheet_name= "Control Group")
print(df1.head())


df2 = pd.read_excel("C:\\Users\\hazal\\Downloads\\AB_TEST.xlsx",sheet_name= "Test Group")
print(df2.head())

print(df1.info())
print(df2.info())
print(df1.describe())
print(df2.describe())
print(df1.isnull().sum())
print(df2.isnull().sum())


print(df1.groupby("Purchase").agg({"Earning":"sum"}).sort_values(by="Purchase",ascending=False))
print(df2.groupby("Purchase").agg({"Earning":"sum"}).sort_values(by="Purchase",ascending=False))
# As a first impression We can't see connection between Purchase and Earning

df3 = pd.concat([df1,df2])
print(df3.count())

print(df1["Purchase"].mean())
print(df2["Purchase"].mean())


# Normallik Varsayımını yapıyoruz.
#H0: Kontrol ve test grupları arasında Purchase açısından istatistiksel olarak anlamlı bir fark yoktur.  
#H1: Kontrol ve test grupları arasında Purchase açısından istatistiksel olarak anlamlı bir fark vardır.
import scipy.stats as stats
purchase_control = df1["Purchase"]
purchase_test = df2["Purchase"]
s_stat,pvalue = stats.ttest_ind(purchase_control,purchase_test,equal_var=False)
print(f"T:Statistic:{s_stat:.4f},p-value:{pvalue:.4f}")
if pvalue<0.05:
  print("H0 Reddedilir.Kontrol ve test grupları arasında Purchase açısından anlamlı bir fark vardır.")
else:
  print("H0 Reddedilmez. Kontrol ve test grupları arasında Purchase açısından anlamlı bir fark yoktur.")
# Burada p-value değerim 0.05 ten büyük p-value:0.3494 yani H0 reddedilmiyor. yani varyans homojenliğine bakabiliriz.


# Normallik için kullanabilecğim bir diğer yöntem Shapiro-Wilk testi
from scipy.stats import shapiro
purchase_control = df1["Purchase"]
purchase_test = df2["Purchase"]
stat_control, p_control = shapiro(purchase_control)
stat_test, p_test = shapiro(purchase_test)
print(f"Shapiro-Wilk Test (Control Grup): pvalue = {p_control:.4f}")
print(f"Shapiro-Wilk Test (Test Grup): pvalue = {p_test:.4f}")
if p_control < 0.05 or p_test < 0.05:
    print("H0 reddedilir. Veri normal dağılmıyor. Bartlett testi yerine Levene testi kullanılmalı.")
else:
    print("H0 reddedilmez. Veri normal dağılıyor. Bartlett testi yada levene testi kullanılabilir.")




# Varyans Homojenliği bakalım.
from scipy.stats import levene
purchase_control = df1["Purchase"]
purchase_test = df2["Purchase"]
stat,pvalue = levene(purchase_control,purchase_test)
print(f"T:Statistic:{stat:.4f},p-value:{pvalue:.4f}")
if pvalue<0.05:
  if p < 0.05:
    print("Varyanslar eşit değil, equal_var=False kullanılmalı")
else:
    print("Varyanslar eşit, equal_var=True kullanılabilir")
# Varyanslar homojen çıktılar. Bağımsız iki örneklem ttesti yani parametrik testi uygulicaz. 



from scipy.stats import ttest_ind

purchase_control = df1["Purchase"]
purchase_test = df2["Purchase"]

# t-testi (Varyanslar eşit olduğu için equal_var=True)
s_stat, pvalue = ttest_ind(purchase_control, purchase_test, equal_var=True)

# Sonuçları yazdır
print(f"T-Statistic: {s_stat:.4f}, p-value: {pvalue:.4f}")

# p-value kontrolü
if pvalue < 0.05:
    print("H0 reddedilir: Kontrol ve test grupları arasında istatistiksel olarak anlamlı bir fark vardır.")
else:
    print("H0 reddedilmez: Kontrol ve test grupları arasında istatistiksel olarak anlamlı bir fark yoktur.")

# p-value değeri 0.3493 yani 0.05 ten büyük, H0 hipotezini reddedemeyiz. Bu iki grup arasında istatistiksel olarak anlamlı bir fark yok.
# Yani, Average Bidding’in Maximum Bidding’den üstün olduğunu söyleyemeyiz.
# Veriyi okuduk ve iki grubun ortalamalarını hesapladık.
# İstatistiksel testlerin gerektirdiği varsayımları kontrol ettik (normallik ve varyans homojenliği).
# Bağımsız İki Örneklem T-Testi uygulayarak iki grubun ortalamalarının farklı olup olmadığını test ettik.
# Sonuç olarak, istatistiksel olarak anlamlı bir fark olmadığı için Maximum Bidding ve Average Bidding arasında bir performans farkı olmadığını belirledik.

















