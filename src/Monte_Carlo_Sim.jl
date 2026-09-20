using Random
using DataFrames

const Arbeit_start = 0.0                        # Arbeitsstart bei 6:45 Uhr = Null
const Arbeit_ende = 8.5 * 60 * 60               # Arbeitsende nach 8,5h
const Frühstückspause_start = 2.5 * 60 * 60     # Frühstückspause 9:15 Uhr
const Frühstückspause_ende = 2.75 * 60 * 60     # Frühstücksende 9:30Uhr
const Frühstückspause = 15 * 60                 # 15min
const Mittagspause_start = 5.75 * 60 * 60       # Mittagspause um 12:30 Uhr
const Mittagspause_ende = 6.25 * 60 * 60        # Mittagspausenende, 12:30 Uhr
const Mittagspause = 30 * 60                    # 30min
const Abfertigung = 0.5 * 60 * 60               # Die Ich-mach-das-noch-fertig-Zeit
const Rüstzeit = 10 * 60                        # 10min
const Bauteilzeit = 3 * 60                      # 3min
const Rüstzeit_Streuung = 0.3                   # 30%-Streuung
const Bauteilzeit_Streuung = 0.3                # 30%-Streuung
const Bauteile_min = 5                          # Mindestens 5 Bauteile Pro Produktionsauftrag
const Bauteile_max = 200                        # Maximal 200 Bauteile Pro Produktionsauftrag

# --- Job-Kontainer ---
@kwdef mutable struct Job  # Großbuchstabe (Konvention für Typen)
    ID :: Int64
    Start :: Float64
    Ende :: Float64
    Bauteile :: Int64
    Bauteilzeit :: Float64  # Hier: Bauteilzeit pro Stück (nicht Gesamtzeit!)
    Rüstzeit :: Float64
    Restzeit :: Float64
    Pause :: Float64
end

function Base.show(io::IO, j::Job)
           println("Job-ID: $(j.ID)\nn: $(j.Bauteile)\nRüstzeit: $(j.Rüstzeit)\nBauteilzeit: $(j.Bauteilzeit)\nStart: $(j.Start)\nEnde: $(j.Ende)\nRestzeit: $(j.Restzeit)\nPause: $(j.Pause)")
end

# --- Job-Generieren ---
function MakeJob(ID :: Int64, Start :: Float64, rng::AbstractRNG)
    n = rng_log(rng, Bauteile_min, Bauteile_max)
    newBauteilzeit = Bauteilzeit * (1.0 + (rand(rng) * 2 - 1) * Bauteilzeit_Streuung)
    newRüstzeit = Rüstzeit * (1.0 + (rand(rng) * 2 - 1) * Rüstzeit_Streuung)
    Dauer = newRüstzeit + n * newBauteilzeit
    # --- Pausen-Check ---
    # Frühstückspause (9:15–9:30)
    Pause = Pausen_Check(Start,Dauer)
    Ende = Start + Dauer + Pause
    Restzeit = Ende - Start
    newJob = Job(ID,Start,Ende,n,newBauteilzeit,newRüstzeit,Restzeit,Pause)
    return newJob
end

# --- Zufällige logarithmische Verteilung der Bauteilanzahl ---
function rng_log(rng::AbstractRNG, min_val::Int, max_val::Int)
    log_min = log(min_val)
    log_max = log(max_val)
    log_rand = rand(rng) * (log_max - log_min) + log_min
    return Int(round(exp(log_rand)))
end

function Pausen_Check(Start::Float64, Dauer::Float64)
    Pause = 0.0

    # Frühstückspause (9:15–9:30)
    if Start < Frühstückspause_start &&
       (Start + Dauer) > Frühstückspause_start
        Pause += Frühstückspause
    end

    # Mittagspause (12:30–13:00)
    if Start < Mittagspause_start &&
       (Start + Dauer) > Mittagspause_start
        Pause += Mittagspause
    end

    return Pause
end

function Montecarlo_Simulation(Iterationen::Int64)
    rng = MersenneTwister(42)

    ID = Vector{Int64}(undef, Iterationen)
    Bauteilanzahl = Vector{Int64}(undef, Iterationen)
    Rüstzeit = Vector{Float64}(undef, Iterationen)
    Bauteilzeit = Vector{Float64}(undef, Iterationen)
    Gesamtzeit = Vector{Float64}(undef, Iterationen)
    Startzeit = Vector{Float64}(undef, Iterationen)
    Endzeit = Vector{Float64}(undef, Iterationen)
    Pause = Vector{Float64}(undef, Iterationen)

    Tag = 1
    aktuelle_Zeit = 0.0
    idx = 1

    for i in 1:Iterationen
        # Neuer Tag nötig?
        if aktuelle_Zeit > Arbeit_ende + Abfertigung
            Tag += 1
            aktuelle_Zeit = 0.0
        end

        myJob = MakeJob(idx, aktuelle_Zeit, rng)

        if myJob.Ende < Arbeit_ende
            # Job wird innerhalb des Tages fertiggestellt
            aktuelle_Zeit += myJob.Restzeit

        elseif myJob.Ende < Arbeit_ende + Abfertigung
            # Job wird noch während der Abfertigungszeit fertig
            aktuelle_Zeit = 0.0
            Tag += 1

        else
            # Job wird nicht fertiggestellt
            myJob.Restzeit = myJob.Ende - Arbeit_ende

            zusätzliche_Pause = Pausen_Check(0.0, myJob.Restzeit)
            myJob.Ende += zusätzliche_Pause

            # Falls die zusätzliche Pause zur Gesamtpause gehören soll:
            myJob.Pause += zusätzliche_Pause

            idx += 1
        end

        # Gemeinsames Abspeichern für alle Fälle
        ID[i] = idx
        Bauteilanzahl[i] = myJob.Bauteile
        Rüstzeit[i] = myJob.Rüstzeit
        Bauteilzeit[i] = myJob.Bauteilzeit
        Gesamtzeit[i] = myJob.Ende - myJob.Start
        Startzeit[i] = myJob.Start
        Endzeit[i] = myJob.Ende
        Pause[i] = myJob.Pause

        # Nächsten Job vorbereiten
        idx += 1
    end

    return DataFrame(
        ID = ID,
        Bauteilanzahl = Bauteilanzahl,
        Rüstzeit = Rüstzeit,
        Bauteilzeit = Bauteilzeit,
        Gesamtzeit = Gesamtzeit,
        Startzeit = Startzeit,
        Endzeit = Endzeit,
        Pausenzeit = Pause
    )
end

df = Montecarlo_Simulation(10_000)

using GLMakie

fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax,df.Bauteilanzahl,df.Gesamtzeit)
fig
