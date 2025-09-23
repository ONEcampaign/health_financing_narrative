from pathlib import Path


class Paths:
    """Class to store the paths to the data and output folders."""

    project = Path(__file__).resolve().parent.parent
    raw_data = project / "raw_data"
    output = project / "output"
    scripts = project / "scripts"

    population = raw_data / "un_population.csv"
    life_expectancy = raw_data / "who_life_expectancy.csv"
    child_mortality = raw_data / "child-mortality-igme.csv"
    gdp_pc = raw_data / "gdp-per-capita-worldbank.csv"
