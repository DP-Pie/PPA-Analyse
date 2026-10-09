"""
# DF_Clean_Up.jl
Dieses Skript bereinigt die Daten der CSV-Datei der PPA-Analyse und speichert diese ab.
Folgende Schritte werden durchgeführt:
1. Die Spalten Bezeichnung1 bis Bezeichnung4 werden zu einer Spalte Bezeichnung zusammengefasst.
2. Die Spalten Isttr und Solltr werden in Sekunden umgerechnet und in den Spalten Isttrsek und Solltrsek gespeichert.
3. Die Spalten Isttr, Solltr, ZEtr und ZEte werden gelöscht.    
"""

using DataFrames
using CSV

"""
# Funktion zum Parsen von Integernwerten
Wandelt einen String in eine ganze Zahl (Integer) um, wobei das deutsche Format (z.B. "1.456,00") berücksichtigt wird.
Leer Strings oder fehlende Werte werden als `missing` zurückgegeben. Ungültige Werte werfen eine ArgumentError.
"""
function parse_menge(x)
    if ismissing(x)
        return missing
    elseif x isa Integer
        return x
    elseif x isa Real
        isinteger(x) || throw(ArgumentError("Keine ganze Zahl: $x"))
        return Int(x)
    end

    s = strip(string(x))

    isempty(s) && return missing

    # Deutsches Format: 1.456,00 -> 1456.00
    s = replace(s, "." => "", "," => ".")

    wert = Int(round(parse(Float64, s)))

    isfinite(wert) || throw(ArgumentError("Ungültiger Wert: $x"))
    isinteger(wert) || throw(ArgumentError("Keine ganze Zahl: $x"))

    return wert
end

"""
# Funktion zum Parsen von Floatwerten
Wandelt einen String in eine Zahl (Floatr) um, wobei das deutsche Format (z.B. "1.456,00") berücksichtigt wird.
Leer Strings oder fehlende Werte werden als `missing` zurückgegeben. Ungültige Werte werfen eine ArgumentError.
"""
function parse_float(x)
    if ismissing(x)
        return missing
    elseif x isa Integer
        return x
    elseif x isa Real
        isinteger(x) || throw(ArgumentError("Keine ganze Zahl: $x"))
        return Int(x)
    end

    s = strip(string(x))

    isempty(s) && return missing

    # Deutsches Format: 1.456,00 -> 1456.00
    s = replace(s, "." => "", "," => ".")

    wert = round(parse(Float64, s), digits=1)

    isfinite(wert) || throw(ArgumentError("Ungültiger Wert: $x"))

    return wert
end

df_path = pick_file(filterlist="*csv")
df = CSV.read(df_path, DataFrame)

df_clean = copy(df)
# Teil von String31 in String convertieren
transform!(df_clean, :Teil => ByRow(string) => :Teil)
# Bezeichnungen zusammenfassen
transform!(
    df_clean,
    [:Bezeichnung1, :Bezeichnung2, :Bezeichnung3, :Bezeichnung4] =>
        ByRow((a, b, c, d) ->
            join(skipmissing((a, b, c, d)), " ")
        ) =>
        :Bezeichnung
)
# Bezeichnung1 bis 4 löschen
select!(df_clean, Not([:Bezeichnung1, :Bezeichnung2, :Bezeichnung3, :Bezeichnung4]))
# Aktivität von String31 zu String konvertieren
transform!(df_clean, :Aktivität => ByRow(string) => :Aktivität)
# von String31 zu Int konvertieren
transform!(df_clean, :rMenge => ByRow(parse_menge) => :rMenge)
transform!(df_clean, :Sollte => ByRow(parse_menge) => :Sollte)
transform!(df_clean, :IstteSek => ByRow(parse_menge) => :IstteSek)
transform!(df_clean, :Isttr => ByRow(parse_float) => :Isttr)
# isttr in Sekunden umrechnen

function in_sekunden(wert, einheit)
    if ismissing(wert)
        return missing
    elseif !ismissing(einheit) &&
           lowercase(strip(string(einheit))) == "min"
        return wert * 60
    else
        return wert
    end
end

transform!(
    df_clean,
    [:Isttr, :ZEtr] => ByRow(in_sekunden) => :Isttrsek
)
transform!(
    df_clean,
    [:Solltr, :ZEtr] => ByRow(in_sekunden) => :Solltrsek
)

select!(df_clean, Not([:Isttr, :Solltr, :ZEtr,:ZEte]))
select!(df_clean, Not([:Istte,:Isttesn,:IstteSollte,:IsttrSolltr]))

select!(df_clean, [:Teil, :TA, :Bezeichnung, :Aktivität, :RMNr, :rMenge, :Sollte, :IstteSek, :Solltrsek, :Isttrsek, :Zustand])

# Float-Werte korrekt anzeigen druch runden.
transform!(df_clean, :tges => ByRow(x -> round(x, digits=1)) => :tges)
transform!(df_clean, :Isttrsek => ByRow(x -> round(x, digits=1)) => :Isttrsek)

# :Zustand von Nein/Ja in Boolean umwandeln.
transform!(
    df_clean,
    :Zustand => ByRow(x ->
        ismissing(x) ? missing :
        lowercase(strip(string(x))) == "ja"
    ) => :Zustand
)

#---------------------------------------------
# !!! ACHTUNG !!! Ab hier wird es unsauber
#---------------------------------------------

#Datenframe aus Selektion erstellen.
teile_df = DataFrame(
    Aktivität = [teil.Aktivität for teil in values(Teiledict)],
    Teil = [teil.name for teil in values(Teiledict)],
    n = [teil.n for teil in values(Teiledict)],
    Bezeichnung = [teil.Bezeichnung for teil in values(Teiledict)]
)
#Datenframe sortieren 
sort!( teile_df, [:Aktivität, :n, :Teil], rev = [false, true, false])

insertcols!( teile_df, 1, :ID => 1:nrow(teile_df))

#leere Spalte für die Gruppierung
insertcols!( teile_df, :Gruppe => Vector{Union{Missing, String}}(missing, nrow(teile_df)))

CSV.write(save_file(), df_clean, writeheader=true)

CSV.write("PPA-BDE_Aktivitäten2.csv", teile_df)

# Schritt zum korrigieren der Aktivitätenbezeichnung 
for i in 1:nrow(teile_df)
    if startswith(teile_df.Aktivität[i],"Prüfung Pro")
        teile_df.Aktivität[i] = "Prüfung/Programmieru"
    end
end

# Df neu ordnen, da einige Aktivitäten ersetzt
sort!( teile_df, [:Aktivität, :n, :Teil], rev = [false, true, false])
teile_df.ID = [ i for i in 1:nrow(teile_df)] #IDs neu vergeben
vscodedisplay(teile_df)

Aktivitäten_df = combine(groupby(teile_df, [:Aktivität]), nrow => :count)
sort!( Aktivitäten_df, [:Aktivität, :count], rev = [false, true])

vscodedisplay(Aktivitäten_df)

# 1. Bereite das Mapping aus teile_df vor
# (Falls in teile_df schon eine Spalte "AktivitätNeu" existiert, ersetze :Aktivität durch :AktivitätNeu)
mapping_df = select(teile_df, :Teil, :Aktivität => :AktivitätNeu, :Gruppe)

# 2. Führe einen LEFT JOIN durch (behält alle Zeilen aus df, fügt Spalten aus teile_df hinzu)
df = leftjoin(df, mapping_df, on = :Teil)

# 3. Verschiebe "AktivitätNeu" direkt hinter "Aktivität"
aktivitaet_idx = findfirst(==(:Aktivität), names(df))
if !isnothing(aktivitaet_idx)
    # Speichere die Spalte
    aktivitaet_neu_data = df.AktivitätNeu
    # Entferne sie
    select!(df, Not(:AktivitätNeu))
    # Füge sie nach "Aktivität" ein
    insertcols!(df, aktivitaet_idx + 1, :AktivitätNeu => aktivitaet_neu_data)
end

CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)

#---------------------------------------------
# Beispiel einer Gruppenbildung mit Längen
#---------------------------------------------
Kabel_df = filter(:Aktivität => a -> a == "Kabelfertigung", Aktivitäten_df)

transform!(Kabel_df, :Gruppe => ByRow(string) => :Gruppe)

for i in 1:nrow(Kabel_df)
    if  contains(Kabel_df.Bezeichnung[i],"Modularkabel")
        Kabel_df.Gruppe[i] = "Modularkabel"
    elseif contains(Kabel_df.Bezeichnung[i],"Flachbandkabel")
        Kabel_df.Gruppe[i] = "Flachbandkabel"
    elseif contains(Kabel_df.Bezeichnung[i],"Einzelleitung")
        Kabel_df.Gruppe[i] = "Einzelleitung"
    else
         Kabel_df.Gruppe[i] = "Diverse-Kabel"
    end
end

update_df_gruppe!(Aktivitäten_df,Kabel_df, gruppe=:Länge, fehlwert = 0.0)

L_Vec = [   90,    1_100,  1_500,  970,    1_150,
            1_500, 3_000,  100,    5_000,  90,
            200,   220,    3_250,  3_250,  80,
            60,    2_000,  2_550,  450,    5_000,
            20_000,155,    1_000,  4_150,  2_500,
            1_700, 3_000,  10_000, 1_000,  0,
            120,   5_000,  195,    195,    95,
            95,    235,    235,    235,    95]
Kabel_df.Länge .= L_Vec

update_df_gruppe!(Aktivitäten_df,Kabel_df)
update_df_gruppe!(Aktivitäten_df,Kabel_df, gruppe=:Länge, fehlwert = 0.0)
CSV.write("PPA-BDE_Aktivitäten2.csv", Aktivitäten_df)

# Update der AktivitätNeu und Gruppe Spalten
df_path = "PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV"
df = CSV.read(df_path, DataFrame)

df.Gruppe .= "missing"
Aktivitäten_df = CSV.read("PPA-BDE_Aktivitäten2.csv", DataFrame)
update_df_gruppe!(df,Aktivitäten_df)
df.AktivitätNeu .= "missing"
Aktivitäten_df.AktivitätNeu .= Aktivitäten_df.Aktivität
update_df_gruppe!(df,Aktivitäten_df,gruppe=:AktivitätNeu)

CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)

#---------------------------------------------
# Liste an Verlinkte Dokumenten einfügen für bessere Gruppierung
#---------------------------------------------
KK_df = CSV.read("20260930_proALPHA AV KK.csv", DataFrame)
KKdict = Dict{String,String}()
for i in 1:nrow(KK_df)
    ismissing(KK_df.Teilenummer[i]) && continue
    link = ismissing(KK_df.Linkname[i]) ? "" : String(KK_df.Linkname[i])
    KKdict[String(KK_df.Teilenummer[i])] = link
end

df.Linkname .= ""
for i in 1:nrow(df)
    df.Linkname[i] = get(KKdict,df.Teil[i],"") #Such den Wert im Dictonary, falls nicht gefunden gebe den Default zurück. 
end

CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)

# Neue Gruppierung durchführen
df.GruppeNeu .= String.(df.Gruppe)

for i in 1:nrow(df)

    # Teile mit Präfix BP,BKK und KK überspringen
    if startswith(df.Teil[i], r"^(BP|BKK|KK)")
        continue
    # Überspringe die Aktivitäten kabelfertigung und Modularkabel prüfen
    elseif df.AktivitätNeu[i] == "Modularkabel prüfen" || df.Aktivität == "Kabelfertigung"
        continue
    # bestimmte Verlinkungen überspringen die mit ... beginnen 
    elseif startswith(df.Linkname[i], r"^(im|auf|mit|EI|DV|MA-Tankgeber|S:\\)")
        continue
    elseif df.AktivitätNeu[i] == "Verpacken"
        continue
    end

    df.GruppeNeu[i] = String(df.Linkname[i])
end

# AlteGruppe verwerfen und neu Gruppe setzen
select!(df, Not(:Gruppe), :GruppeNeu => :Gruppe)
CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)

# Missing in den Einträgen ersetzen 
df_path = "\\\\FIS-VTL-01\\Userdirs\\d.pietzner\\Documents\\Programmieren\\BDE-Analyse_2026\\PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV"
df = CSV.read(df_path, DataFrame)
string_cols = [c for c in names(df) if eltype(df[!, c]) >: Missing && nonmissingtype(eltype(df[!, c])) <: AbstractString]
transform!(df, string_cols .=> ByRow(x -> coalesce(x, "")) .=> string_cols)
CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)