function Zeitstempel(Zeit::Float64, Basiszeit::Time=Time(6,45,0))
    if Zeit < 0 then
        throw(ArgumentError("Zeit darf nicht negativ sein! Zeit: $Zeit"))
    end
    Zeitstempel = Basiszeit + Second(round(Int64,Zeit))
    return Zeitstempel
end

# --- NEUE ZEITSTEMPEL-FUNKTION (unterstützt mehrere Tage) ---
function Zeitstempel(Zeit::Float64, Tag::Int64)
    Basiszeit = Time(6, 45, 0)  # Nur die Uhrzeit, kein Datum
    return Basiszeit + Second(round(Int64, Zeit))  # → gibt Time zurück
end

# --- Job-Generieren (Hauptversion mit Parametern) ---
function MakeJob(ID::Int64, Start::Float64, rng::AbstractRNG,
                 te::Float64, tr::Float64,
                 te_streuung::Float64=0.3,
                 tr_streuung::Float64=0.3,
                 bauteile_min::Int64=2,
                 bauteile_max::Int64=100)
    n = rng_decay(rng, bauteile_min, bauteile_max)
    newBauteilzeit = te * (1.0 + (rand(rng) * 2 - 1) * te_streuung)
    newRüstzeit = tr * (1.0 + (rand(rng) * 2 - 1) * tr_streuung)
    Dauer = newRüstzeit + n * newBauteilzeit
    # --- Pausen-Check ---
    Pause = Pausen_Check(Start, Dauer)
    Ende = Start + Dauer + Pause
    Restzeit = Ende - Start
    Gesamtzeit = Ende - Start
    return Job(ID, Start, Ende, n, newBauteilzeit, newRüstzeit, Restzeit, Pause, Gesamtzeit)
end

# --- Legacy-Wrapper für alte Aufrufe ---
MakeJob(ID::Int64, Start::Float64, rng::AbstractRNG) =
    MakeJob(ID, Start, rng, MCconst.Bauteilzeit, MCconst.Rüstzeit, 0.3, 0.3, 2, 100)

# --- Zufällige logarithmische Verteilung der Bauteilanzahl ---
function rng_log(rng::AbstractRNG, min_val::Int, max_val::Int)
    log_min = log(min_val)
    log_max = log(max_val)
    log_rand = rand(rng) * (log_max - log_min) + log_min
    return Int(round(exp(log_rand)))
end

# --- Angefittet aus der Verteilungsfunktion alle TE-Einträge
function rng_decay(rng::AbstractRNG, min_val::Int, max_val::Int)
    x_values = min_val:max_val
    weights = [0.133 * exp(-0.125 * x) + 0.001 for x in x_values]
    total = sum(weights)
    u = rand(rng) * total
    cumulative = 0.0
    for (i, w) in enumerate(weights)
        cumulative += w
        if u <= cumulative
            return x_values[i]
        end
    end
    return x_values[end]
end

# --- Pausenzeiten einrechnen ---
function Pausen_Check(Start::Float64, Dauer::Float64)
    Pause = 0.0
    # Frühstückspause (9:15–9:30)
    if Start < MCconst.Frühstückspause_start && (Start + Dauer) > MCconst.Frühstückspause_start
        Pause += MCconst.Frühstückspause
    end
    # Mittagspause (12:30–13:00)
    if Start < MCconst.Mittagspause_start && (Start + Dauer) > MCconst.Mittagspause_start
        Pause += MCconst.Mittagspause
    end
    return Pause
end

# --- Hauptversion: Montecarlo_Simulation mit Parametern ---
function Montecarlo_Simulation(Iterationen::Int64, rng::AbstractRNG, start_idx::Int64,
                                te::Float64, tr::Float64)
    # Iterationsspeicher der Metadaten vorbereiten
    ID = Vector{Int64}(undef, Iterationen)
    Bauteilanzahl = Vector{Int64}(undef, Iterationen)
    Rüstzeit = Vector{Float64}(undef, Iterationen)
    Bauteilzeit = Vector{Float64}(undef, Iterationen)
    Gesamtzeit = Vector{Float64}(undef, Iterationen)
    Startzeit = Vector{Float64}(undef, Iterationen)
    Startuhrzeit = Vector{Time}(undef, Iterationen)
    Endzeit = Vector{Float64}(undef, Iterationen)
    Enduhrzeit = Vector{Time}(undef, Iterationen)
    Pause = Vector{Float64}(undef, Iterationen)
    Tage = Vector{Int64}(undef, Iterationen)

    # Startbedingungen
    Tag = 1
    aktuelle_Zeit = 0.0
    idx = start_idx

    # Simulationsschleife
    for i in 1:Iterationen
        # Neuer Tag nötig?
        if aktuelle_Zeit > MCconst.Arbeit_ende + MCconst.Abfertigung
            Tag += 1
            aktuelle_Zeit = 0.0
        end

        # job/Produktionsauftrag erstellen
        myJob = MakeJob(idx, aktuelle_Zeit, rng, te, tr)

        if myJob.Ende < MCconst.Arbeit_ende
            # Job wird innerhalb des Tages fertiggestellt
            aktuelle_Zeit += myJob.Restzeit
        elseif myJob.Ende < MCconst.Arbeit_ende + MCconst.Abfertigung
            # Job wird noch während der Abfertigungszeit fertig
            aktuelle_Zeit = 0.0
            Tag += 1
        else
            # Job wird nicht fertiggestellt
            myJob.Restzeit = myJob.Ende - MCconst.Arbeit_ende
            zusätzliche_Pause = Pausen_Check(0.0, myJob.Restzeit)
            myJob.Ende += zusätzliche_Pause
            myJob.Pause += zusätzliche_Pause
            myJob.Gesamtzeit += zusätzliche_Pause
            idx += 1
        end

        # Abspeichern
        ID[i] = idx
        Bauteilanzahl[i] = myJob.Bauteile
        Rüstzeit[i] = myJob.Rüstzeit
        Bauteilzeit[i] = myJob.Bauteilzeit
        Gesamtzeit[i] = myJob.Gesamtzeit
        Startzeit[i] = myJob.Start
        Startuhrzeit[i] = Zeitstempel(myJob.Start, Tag)
        Endzeit[i] = myJob.Ende
        Enduhrzeit[i] = Zeitstempel(myJob.Ende, Tag)
        Pause[i] = myJob.Pause
        Tage[i] = Tag

        # Nächsten Job vorbereiten
        idx += 1
    end

    # Ausgabe
    return DataFrame(
        ID = ID,
        Bauteilanzahl = Bauteilanzahl,
        Rüstzeit = Rüstzeit,
        Bauteilzeit = Bauteilzeit,
        Gesamtzeit = Gesamtzeit,
        Startzeit = Startzeit,
        Endzeit = Endzeit,
        Pausenzeit = Pause,
        Startuhrzeit = Startuhrzeit,
        Enduhrzeit = Enduhrzeit,
        Tag = Tage
    )
end

# --- Legacy-Wrapper ---
Montecarlo_Simulation(Iterationen::Int64) =
    Montecarlo_Simulation(Iterationen, Random.default_rng(), 1, MCconst.Bauteilzeit, MCconst.Rüstzeit)

Montecarlo_Simulation(Iterationen::Int64, rng::AbstractRNG, start_idx::Int64) =
    Montecarlo_Simulation(Iterationen, rng, start_idx, MCconst.Bauteilzeit, MCconst.Rüstzeit)

# --- Beschleunigung der berechnung durch Multithreading
function parallel_montecarlo(total_iterations::Int64, n_threads::Int64,
                           te::Float64, tr::Float64) 
    iterations_per_thread = cld(total_iterations, n_threads)
    results = Vector{DataFrame}(undef, n_threads)

    Threads.@threads for tid in 1:n_threads
        rng = MersenneTwister(tid)
        start_idx = (tid - 1) * iterations_per_thread + 1
        its = (tid == n_threads) ? (total_iterations - (n_threads - 1) * iterations_per_thread) : iterations_per_thread
        results[tid] = Montecarlo_Simulation(its, rng, start_idx, te, tr)  # te/tr übergeben
    end

    # Ergebnisse kombinieren und sortieren
    combined = vcat(results...)
    sort!(combined, :Startzeit)

    # Tage neu berechnen
    combined.Tag .= 1
    current_tag = 1
    for i in 2:nrow(combined)
        if combined.Endzeit[i] > MCconst.Arbeit_ende + MCconst.Abfertigung
            current_tag += 1
        end
        combined.Tag[i] = current_tag
    end
    return combined
end

function parallel_montecarlo(total_iterations::Int64, n_threads::Int64)
    return  parallel_montecarlo(total_iterations::Int64, n_threads::Int64, MCconst.Bauteilzeit, MCconst.Rüstzeit)
end

# --- Hauptversion: MonteCarlo_in_da_Lopp mit Parametern ---
function MonteCarlo_in_da_Lopp(Iterationen_Außen::Int64, Iterationen_Innen::Int64,
                               te::Float64, tr::Float64)
    result = Matrix{Float64}(undef, Iterationen_Außen, 2)

    Threads.@threads for i in 1:Iterationen_Außen
        rng = MersenneTwister(i)  # Thread-lokaler RNG
        tmp_DF = Montecarlo_Simulation(Iterationen_Innen, rng, 1, te, tr)
        Reg_DF = DataFrame(x = tmp_DF.Bauteilanzahl, y = tmp_DF.Gesamtzeit)
        Reg_M = RobustModels.rlm(
            @formula(y ~ x),
            Reg_DF,
            RobustModels.MMEstimator{RobustModels.TukeyLoss}();
            σ0 = :mad,
            maxiter = 1000
        )
        Coef = StatsAPI.coef(Reg_M)
        result[i,:] .= Coef  # ✅ Broadcasting!
    end

    return result
end

# --- Legacy-Wrapper ---
MonteCarlo_in_da_Lopp(Iterationen_Außen::Int64, Iterationen_Innen::Int64) =
    MonteCarlo_in_da_Lopp(Iterationen_Außen, Iterationen_Innen, MCconst.Bauteilzeit, MCconst.Rüstzeit)

"""
 #  
 Hilfsfunktion zur Übersicht der Metadaten und ihren Statistiken.
 :mean, :std, :min, :q05, :q25, :median, :q75, :q95, :max, :schiefe, :qdk

 ## Beispiel:

 ```Julia
 # Alle Statistiken (Standard)
 stats_all = DF_Stats(df)

 # Nur Mittelwert und Standardabweichung
 stats_mean_std = DF_Stats(df, [:mean, :std])

 # Nur Median
 stats_median = DF_Stats(df, :median)

 # Mittelwert, Median und Max
 stats_custom = DF_Stats(df, [:mean, :median, :max])
 ```

"""
function DF_Stats(df::DataFrame, stats::Union{Vector{Symbol},Symbol,Nothing}=nothing)
    # Filter Time-Spalten heraus
    df = select(df, Not([col for col in names(df) if eltype(df[!, col]) == Time]))
    
    # Standardmäßig alle Statistiken berechnen
    if stats === nothing
        stats = [:mean, :std, :min, :q05, :q25, :median, :q75, :q95, :max, :schiefe, :qdk]
    elseif stats isa Symbol
        stats = [stats]  # Einzelsymbol in Vektor umwandeln
    end

    # Ergebnisse-Dict
    result = Dict{Symbol,Vector}()
    names_df = names(df)

    # Einfache Statistiken (können direkt berechnet werden)
    simple_stats = Dict(
        :mean => (col -> mean(col), :Mittelwert),
        :std => (col -> std(col), :Standardabweichung),
        :min => (col -> minimum(col), :Min),
        :q05 => (col -> quantile(col, 0.05), :Q05),
        :q25 => (col -> quantile(col, 0.25), :Q25),
        :median => (col -> quantile(col, 0.5), :Median),
        :q75 => (col -> quantile(col, 0.75), :Q75),
        :q95 => (col -> quantile(col, 0.95), :Q95),
        :max => (col -> maximum(col), :Max)
    )

    # Berechne einfache Statistiken
    for stat in stats
        if haskey(simple_stats, stat)
            func, colname = simple_stats[stat]
            result[colname] = [func(col) for col in eachcol(df)]
        end
    end

    # Spezielle Statistiken (benötigen andere Werte)
    if :schiefe in stats
        mean_vals = [mean(col) for col in eachcol(df)]
        median_vals = [quantile(col, 0.5) for col in eachcol(df)]
        result[:Schiefe] = [mean_vals[i] / median_vals[i] for i in eachindex(mean_vals)]
    end

    if :qdk in stats
        q25_vals = [quantile(col, 0.25) for col in eachcol(df)]
        median_vals = [quantile(col, 0.5) for col in eachcol(df)]
        q75_vals = [quantile(col, 0.75) for col in eachcol(df)]
        result[:QDK] = QDK.(q25_vals, median_vals, q75_vals)
    end

    # DataFrame erstellen
    final_df = DataFrame(Variable = names_df)
    for (colname, values) in result
        final_df[!, colname] = values
    end

    return final_df
end

# Legacy-Wrapper für Kompatibilität
DF_Stats(df::DataFrame) = DF_Stats(df, nothing)

# --- Korrigierte Version: Gibt true zurück, wenn EIN Element negativ ist ---
function Vec_Elemen_positiv(vec::Vector{Float64})
    return any(x -> x < 0, vec)  # ✅ Korrekt: true wenn mind. ein Element < 0
end

# --- Hauptfunktion: MonteCarlo_Sim_Grid ---
function MonteCarlo_Sim_Grid(TEs::Vector{Int64}, TRs::Vector{Int64}, n_Pkts::Int64=10, sim_Pkts::Int64=1000)
    return MonteCarlo_Sim_Grid(Float64.(TEs), Float64.(TRs), n_Pkts, sim_Pkts)
end

function MonteCarlo_Sim_Grid(TEs::Vector{Float64}, TRs::Vector{Float64}, n_Pkts::Int64=10, sim_Pkts::Int64=1000)
    if Vec_Elemen_positiv(TEs) || Vec_Elemen_positiv(TRs)
        throw(ArgumentError("Alle Elemente der Inputvektoren müssen >= Null sein!"))
    end

    # 1. Vektoren vorallokieren
    total_rows = length(TEs) * length(TRs) * n_Pkts
    bauteilzeiten = Vector{Float64}(undef, total_rows)
    rüstzeiten = Vector{Float64}(undef, total_rows)
    steigungen = Vector{Float64}(undef, total_rows)
    achsenabschnitte = Vector{Float64}(undef, total_rows)
    te_fehler = Vector{Float64}(undef, total_rows)
    tr_fehler = Vector{Float64}(undef, total_rows)

    # 2. Index für das Befüllen der Vektoren
    idx = 1
    for te_now in TEs
        for tr_now in TRs
            # Keine Änderungen an MCconst mehr! Parameter werden übergeben
            coef_matrix = MonteCarlo_in_da_Lopp(n_Pkts, sim_Pkts, te_now, tr_now)
            for i in 1:n_Pkts
                bauteilzeiten[idx] = te_now
                rüstzeiten[idx] = tr_now
                steigungen[idx] = coef_matrix[i, 1]
                achsenabschnitte[idx] = coef_matrix[i, 2]
                te_fehler[idx] = steigungen[idx] - te_now
                tr_fehler[idx] = achsenabschnitte[idx] - tr_now
                idx += 1
            end
        end
    end

    # 3. Einen DataFrame am Ende erstellen
    return DataFrame(
        Bauteilzeit = bauteilzeiten,
        Rüstzeit = rüstzeiten,
        Steigung = steigungen,
        Achsenabschnitt = achsenabschnitte,
        TE_Fehler = te_fehler,
        TR_Fehler = tr_fehler
    )
end
