"""
# gruppe_aus_teil(teilname; usePrefix::String)

Extrahiert eine Gruppenkennung nach der Baugruppenfamilie aus einem Teilnamen.
Selektiert wird per angegebenen Präfix (`usePräfix`).
Die letzten vier Ziffern bilden die Baugruppenfamilie (mit führenden Nullen aufgefüllt).
Teilnamen ohne passenden Präfix oder ohne abschließende Ziffern werden als `missing` zurückgegeben.
Ein abschließendes 'T' wird vor der Verarbeitung entfernt.
"""
function gruppe_aus_teil(teilname; usePräfix::String)
    if ismissing(teilname)
        return missing
    end

    teilname = string(teilname)

    # teilnamee ignorieren, die mit zwei Buchstaben beginnen
    if !startswith(teilname,usePräfix)
        return missing
    end

    # Abschließendes T entfernen
    teilname = replace(teilname, r"T$" => "")

    # Ziffern am Ende suchen
    match_result = match(r"(\d+)$", teilname)

    # Keine Endziffern vorhanden
    isnothing(match_result) && return missing

    ziffern = match_result.captures[1]

    # Die letzten vier Ziffern verwenden und links mit Nullen auffüllen
    return usePrefix * lpad(last(ziffern, min(4, length(ziffern))), 4, '0')
end

"""
# gruppe_aus_teil(teilname)

Extrahiert eine Gruppenkennung nach der Baugruppenfamilie aus einem Teilnamen.
(Ohne Präfix)
Die letzten vier Ziffern bilden die Baugruppenfamilie (mit führenden Nullen aufgefüllt).
Teilnamen ohne passenden Präfix oder ohne abschließende Ziffern werden als `missing` zurückgegeben.
Ein abschließendes 'T' wird vor der Verarbeitung entfernt.
"""
function gruppe_aus_teil(teilname)
    if ismissing(teilname)
        return missing
    end

    teilname = string(teilname)

    # teilnamee ignorieren, die mit zwei Buchstaben beginnen
    if occursin(r"^[A-Za-z]{2}", teilname)
        return missing
    end

    # Abschließendes T entfernen
    teilname = replace(teilname, r"T$" => "")

    # Ziffern am Ende suchen
    match_result = match(r"(\d+)$", teilname)

    # Keine Endziffern vorhanden
    isnothing(match_result) && return missing

    ziffern = match_result.captures[1]

    # Die letzten vier Ziffern verwenden und links mit Nullen auffüllen
    return lpad(last(ziffern, min(4, length(ziffern))), 4, '0')
end

"""
# gebe_aus_MZ(Bezeichnung::String)

Extrahiert den MZ-Code der Montagezeichnung aus einer Bauteilbezeichnung anhand des Musters `MZ-XXX-YYY`.
Gibt den gefundenen Code zurück oder `"missing"`, falls kein Muster gefunden wurde.
"""
function gebe_aus_MZ(Bezeichnung::String)
    matchresult = match(r"MZ-([^-]+)-([^-]+?)(?=[ ,.!?]|$)",Bezeichnung) #([^-]+)-([^-]+?)(?=[.,!?])($)

    if matchresult !== nothing
        return matchresult.match
    else
        return "missing"
    end

end

"""
 # Gruppe_mit_Länge_erstellen(bezeichnung::String, trigger::String, start_marker::String, end_marker::String; must_match::Bool=false)

 Sucht in einer Bauteilbezeichnung nach einem trigger-String. Wenn dieser vorhanden ist, dann wird nach einem Wert zwischen zwei Markern gesucht.
 Über must_match kann zwischen einem stillen (return: nothing) und einem lauten (Fehlermeldung) Rückmeldung geschaltet werden.
 
 ## Argumente

 - bezeichnung::String    ... Zu durchsuchender String
 - trigger::String        ... Suchstring 
 - start_marker::String   ... Startmarkierung
 - end_marker::String     ... Endmarkierung
 
 [optional]
 - must_match::Bool=false ... Rückmeldung false = nothing, true = error

 ## Beispiel

 ```Julia
 str = "Schlauch PTFE zugeschnitten D=3,8 d=3,2 s=0,3mm Länge=765mm, 30° schräg natur"
 Gruppe_mit_Länge_erstellen(str,"PTFE","Länge=","mm")
 ```

 Ausgabe: `"`PTFE_765`"`

"""
function Gruppe_mit_Länge_erstellen(bezeichnung::String,
                                    trigger::String,
                                    start_marker::String,
                                    end_marker::String;
                                    must_match::Bool=false)
    # Escape spezielle Regex-Zeichen im Trigger/Markern
    trigger_esc = replace(trigger, r"([\.\+\*\?\(\)\[\]\{\}\|\\])" => s"\\\1")
    start_esc = replace(start_marker, r"([\.\+\*\?\(\)\[\]\{\}\|\\])" => s"\\\1")
    end_esc = replace(end_marker, r"([\.\+\*\?\(\)\[\]\{\}\|\\])" => s"\\\1")

    # Regex: Trigger → beliebig → Start-Marker → Zahl → beliebig → End-Marker
    pattern = Regex("(?i)" * trigger_esc * ".*?" * start_esc * "\\s*([0-9]+(?:[.,][0-9]+)?)\\s*" * end_esc)
    matchresult = match(pattern, bezeichnung)
    matchresult === nothing && (must_match ? error("Kein Treffer für $trigger/$start_marker/$end_marker") : return nothing)

    wert = replace(matchresult[1], "," => ".")  # Komma → Punkt für Float
    return trigger * "_" *wert
end

function gruppe_setzen!( df::DataFrame, von::Int, bis::Int, gruppe::String)
    df[
        (df.ID .>= von) .& (df.ID .<= bis),
        :Gruppe
    ] .= gruppe

    return df
end

"""
 # update_df_gruppe!()
 
 Genutzt zum Update der Gruppenspalte des Dataframes "ziel"" mit dem Dataframe "quelle"
 Achtung Funktion verändert den ersten Datensatz "ziel"
 
 ## Beispiel:
 
 alternativen Spalten nutzung, Ersetzen der Länge.
 ```Julia
 update_df_gruppe!(Aktivitäten_df,PTFE_df, gruppe=:Länge, fehlwert = 0.0)
 ```

"""
function update_df_gruppe!(ziel::DataFrame, quelle::DataFrame;
                           on::Symbol = :Teil,
                           gruppe::Symbol = :Gruppe,
                           fehlwert = "missing")
    # Spalten müssen in beiden Frames vorhanden sein
    (hasproperty(ziel, on) && hasproperty(ziel, gruppe)) ||
        throw(ArgumentError("ziel braucht die Spalten $on und $gruppe"))
    (hasproperty(quelle, on) && hasproperty(quelle, gruppe)) ||
        throw(ArgumentError("quelle braucht die Spalten $on und $gruppe"))

    # Nachschlagetabelle, gegen doppelte Schlüssel abgesichert
    lookup = unique(quelle[:, [on, gruppe]], on)

    # Join über temporäre Spalte
    leftjoin!(ziel, rename(lookup, gruppe => :Gruppe_neu); on = on)

    # Ersetze nur dort, wo ein Treffer existiert und der alte Wert "fehlwert" ist
    treffer = (.!ismissing.(ziel.Gruppe_neu)) .& isequal.(ziel[!, gruppe], fehlwert)
    ziel[!, gruppe][treffer] .= ziel.Gruppe_neu[treffer]

    select!(ziel, Not(:Gruppe_neu))
    return nothing
end