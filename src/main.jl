using Pkg

Pkg.add("RobustModels")
Pkg.add("LinRegOutliers")
Pkg.add("JLD2")

using NativeFileDialog
using RobustModels
using LinRegOutliers
using DataFrames
using CSV
using GLMakie
using JLD2

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

# Daten Aufräumen
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

transform!(
    df_clean,
    [:IstteSek, :Isttrsek] =>
        ByRow((a, b) ->
            ismissing(a) || ismissing(b) ? missing : round(a + b,1)
        ) =>
        :tges
) 

CSV.write(save_file(), df_clean, writeheader=true)
