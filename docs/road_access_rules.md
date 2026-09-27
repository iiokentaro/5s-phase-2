# OSM way access rules: Maharashtra and Thailand

Resolved `legal_*` value per `highway=*` class (country default overlay applied onto the shared base), as used by `src/road_access.py` / `src/road_access_rules.json`. A value here is what a way of that class gets *absent any access tag*; an explicit tag on the way (`access`, `vehicle`, `motor_vehicle`, `motorcycle`, `bicycle`, `foot`) overrides it per `access_hierarchy`.

These values feed the variables `A_vru` and `M` of the V_safe rule table in `docs/v_safe_raise_conditions.md`.

### Maharashtra

| highway | motor_vehicle | motorcycle | bicycle | foot |
|---|---|---|---|---|
| `motorway` | yes | no | no | no |
| `motorway_link` | yes | no | no | no |
| `trunk` | yes | yes | yes | yes |
| `trunk_link` | yes | yes | yes | yes |
| `primary` | yes | yes | yes | yes |
| `primary_link` | yes | yes | yes | yes |
| `secondary` | yes | yes | yes | yes |
| `secondary_link` | yes | yes | yes | yes |
| `tertiary` | yes | yes | yes | yes |
| `tertiary_link` | yes | yes | yes | yes |
| `unclassified` | yes | yes | yes | yes |
| `residential` | yes | yes | yes | yes |
| `living_street` | yes | yes | yes | yes |
| `service` | yes | yes | yes | yes |
| `road` | unknown | unknown | unknown | unknown |
| `busway` | restricted | no | no | no |
| `track` | **yes** | yes | yes | yes |
| `path` | no | unknown | yes | yes |
| `footway` | no | no | no | yes |
| `pedestrian` | restricted | no | unknown | yes |
| `cycleway` | no | no | yes | unknown |
| `bridleway` | no | no | unknown | unknown |
| `steps` | no | no | no | yes |
| `crossing` | no | no | unknown | yes |
| `corridor` | no | no | no | yes |
| `construction` | no | no | no | no |
| `proposed` | no | no | no | no |
| `planned` | no | no | no | no |
| `raceway` | no | no | no | no |
| `escape` | restricted | no | no | no |
| `rest_area` | yes | yes | yes | yes |
| `services` | yes | yes | yes | yes |
| `bus_stop` | unknown | unknown | unknown | yes |
| `elevator` | no | no | unknown | yes |
| `via_ferrata` | no | no | no | yes |

Bold = overridden from the shared base for this country.

### Thailand

| highway | motor_vehicle | motorcycle | bicycle | foot |
|---|---|---|---|---|
| `motorway` | yes | no | no | no |
| `motorway_link` | yes | no | no | no |
| `trunk` | yes | yes | yes | yes |
| `trunk_link` | yes | yes | yes | yes |
| `primary` | yes | yes | yes | yes |
| `primary_link` | yes | yes | yes | yes |
| `secondary` | yes | yes | yes | yes |
| `secondary_link` | yes | yes | yes | yes |
| `tertiary` | yes | yes | yes | yes |
| `tertiary_link` | yes | yes | yes | yes |
| `unclassified` | yes | yes | yes | yes |
| `residential` | yes | yes | yes | yes |
| `living_street` | yes | yes | yes | yes |
| `service` | yes | **yes** | yes | **yes** |
| `road` | unknown | unknown | unknown | unknown |
| `busway` | restricted | no | no | no |
| `track` | **yes** | yes | yes | yes |
| `path` | no | **yes** | yes | yes |
| `footway` | no | no | no | yes |
| `pedestrian` | restricted | no | unknown | yes |
| `cycleway` | no | no | yes | unknown |
| `bridleway` | no | no | unknown | unknown |
| `steps` | no | no | no | yes |
| `crossing` | no | no | unknown | yes |
| `corridor` | no | no | no | yes |
| `construction` | no | no | no | no |
| `proposed` | no | no | no | no |
| `planned` | no | no | no | no |
| `raceway` | no | no | no | no |
| `escape` | restricted | no | no | no |
| `rest_area` | yes | yes | yes | yes |
| `services` | yes | yes | yes | yes |
| `bus_stop` | unknown | unknown | unknown | yes |
| `elevator` | no | no | unknown | yes |
| `via_ferrata` | no | no | no | yes |

Bold = overridden from the shared base for this country.

