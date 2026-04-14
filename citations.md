# Parameter provenance and citations for the paper release

This note collects the main external references used by the paper-facing code and manuscript.
It is intentionally pragmatic: some parameters are directly source-backed, while others remain explicit modelling choices calibrated within published ranges.

## Data sources

- **UK reported COVID-19 cases (historic archive):** UKHSA dashboard archive data download page. Used as the public comparison series for daily cases.
  - https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download
- **Google Community Mobility Reports:** historical Google mobility data used for optional workplace / retail / transit multipliers. Google states that updates stopped on **2022-10-15** and that historical data remain publicly available.
  - https://www.google.com/covid19/mobility/
- **OpenStreetMap licensing:** OSM data are used under the Open Database License (ODbL); attribution should remain “© OpenStreetMap contributors”.
  - https://wiki.openstreetmap.org/wiki/Open_Database_License

## Household and residential assumptions

- **Average UK household size = 2.35 (2024):** Office for National Statistics, *Families and households in the UK: 2024*.
  - https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024
- **Typical dwelling / flat sizes:** English Housing Survey headline and floor-space reports are used as anchors for the fallback residential area assumptions.
  - Headline report (average dwelling size around 94 m²):
    - https://assets.publishing.service.gov.uk/media/5e2873b0e5274a6c43d73ca0/2018-19_EHS_Headline_Report.pdf
  - Floor space technical report (flat-size distributions used as motivation for 58–65 m² flat heuristics):
    - https://assets.publishing.service.gov.uk/media/5b45fd6ee5274a3773e665a3/Floor_Space_in_English_Homes_technical_report.pdf

## Schools and education

- **Mainstream school area guidance (BB103):** DfE guidance used as the main anchor for school-space plausibility.
  - https://www.gov.uk/government/publications/area-guidelines-and-net-capacity
  - PDF: https://assets.publishing.service.gov.uk/media/5f23ec238fa8f57acac33720/BB103_Area_Guidelines_for_Mainstream_Schools.pdf
- **Secondary pupil-teacher ratio (~16.7 in 2024/25):** DfE / Explore Education Statistics, *School workforce in England, reporting year 2024*.
  - https://explore-education-statistics.service.gov.uk/find-statistics/school-workforce-in-england/2024
- **Early-years staff-child ratios (1:8 / 1:13 settings referenced in the manuscript):** EYFS framework / explanatory guidance.
  - GOV.UK explainer: https://www.gov.uk/government/publications/early-years-qualifications-and-ratios/early-years-qualifications-and-ratios
  - Ofsted explainer: https://earlyyears.blog.gov.uk/2023/04/20/how-staff-to-child-ratios-work/

## Device-bearing correction for “stayers”

- **Child mobile-phone ownership:** Ofcom, *Children and Parents: Media Use and Attitudes Report 2024* (“By the age of 11, nine in ten children own their own mobile phone”).
  - https://www.ofcom.org.uk/siteassets/resources/documents/research-and-data/media-literacy-research/children/children-media-use-and-attitudes-2024/childrens-media-literacy-report-2024.pdf?v=368229
- **Population aged 0–15:** ONS local indicator used as support for the statement that school-age children are a minority share of the total population.
  - https://www.ons.gov.uk/explore-local-statistics/indicators/percentage-of-the-population-aged-0-to-15

## Offices and workplaces

- **Office density:** BCO material is used as a high-level plausibility anchor. The code’s 14 m²/employee rule is a modelling simplification that sits near commonly cited modern occupancy-density figures.
  - BCO note on post-pandemic effective density (~15 m² per occupant):
    - https://www.bco.org.uk/bco-report-calls-for-new-approach-to-space-planning-as-office-use-reaches-critical-shift
- The exact **14 m²** office rule in code should still be understood as a pragmatic modelling assumption rather than a directly estimated quantity.

## Vaccination / reinfection notes used in code comments

- **COVID-19 vaccine surveillance reports:** UKHSA/PHE surveillance reports used as broad anchors for Delta-era vaccine effectiveness assumptions.
  - Landing page: https://www.gov.uk/government/publications/covid-19-vaccine-weekly-surveillance-reports
  - Example report: https://assets.publishing.service.gov.uk/media/613a32548fa8f503c320a05c/Vaccine_surveillance_report_-_week_36.pdf
- **Transmission reduction after vaccination:** Eyre et al., NEJM 2022.
  - https://www.nejm.org/doi/full/10.1056/NEJMoa2116597
- **90-day reinfection convention:** public-health surveillance definition used in UK reporting.
  - https://www.gov.uk/government/news/new-national-surveillance-of-possible-covid-19-reinfection-published-by-phe
  - https://ukhsa.blog.gov.uk/2022/02/04/changing-the-covid-19-case-definition/

## Parameters that remain explicit modelling choices

The following are still best treated as transparent modelling assumptions rather than sourced constants:

- `CITY_TO_EPS` tolerances in the OD solver.
- The `0.083` correction applied to same-region “stayers”.
- Shop / restaurant employee caps from floor area.
- City-specific non-household contact caps and event-pulse magnitudes.
- Some Alpha / Delta ascertainment schedules and short city-specific multipliers.

These are now left in code as explicit assumptions and should be described that way in the manuscript and supplementary material.
