# Plot-Display erweitern um Residuen-Plot QQ-Plot ...
vscodedisplay(modell_robust_df)
vscodedisplay(teile_df)

Gruppe_mit_Länge_erstellen(teile_df.Bezeichnung[1720],"PTFE","Länge=","mm")

Gruppe_Schlauch_länge(missing_df.Bezeichnung[2])
missing_df.Bezeichnung[1]
vscodedisplay(missing_df)

Print_Regression(Teiledict["KK-SK343-75-AA-002"],1)

# QQ-plot einbinden
fig = Figure()
ax = Axis(fig[1:2,1:2])
bx = Axis(fig[3,3])
res = Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"].points.residuals
qqnorm!(ax,res,qqline = :fit)
res_sorted = sort(res;rev=true)
scatter!(bx, res_sorted)
fig

liste_regressionsmodelle(Teiledict["KK-SK343-75-AA-002"]; returnText = true)



