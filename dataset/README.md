# Mine vision dataset

Recommended starting classes for the SIH prototype:

- `person`
- `helmet`
- `no_helmet`
- `safety_vest`
- `no_vest`
- `fallen_person`
- `smoke`
- `fire`
- `obstacle`
- `blocked_exit`

Use real, permissioned underground-mine imagery wherever possible. Split by
scene/location, not only by random frames, to reduce leakage between train and
validation. Keep the validation set representative of low light, dust,
occlusion and different camera viewpoints.

The app maps only explicitly named classes to safety events. It never infers
"no helmet" from a person who simply lacks a helmet detection.
