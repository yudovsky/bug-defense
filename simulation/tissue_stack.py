"""
Biophotonic Tissue Stack Module.
Defines the layered beetle body and evaluates chromophore-decomposed absorption [mu_a],
scattering [mu_s, mu_s', g], refractive index and heat capacity for each layer.
"""

import os
import sys
import math

# Import invertebrate tissue optics closure package
TISSUE_OPTICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge', 'invertebrate_tissue_optics')
if os.path.isdir(TISSUE_OPTICS_DIR) and TISSUE_OPTICS_DIR not in sys.path:
    sys.path.insert(0, TISSUE_OPTICS_DIR)

try:
    from insect_tissue_optics.layers import Layer as TissueLayer
    HAS_TISSUE_OPTICS = True
except Exception as e:
    HAS_TISSUE_OPTICS = False
    print(f"Warning: Could not import insect_tissue_optics: {e}")

# Armored beetle body used by the 3D runners: dorsal shield, subelytral air cavity, viscera, ventral shield
BASE_LAYERS = [
    # Dorsal Protective Shield (0 to 0.162 mm)
    {'name': 'epicuticle_wax', 'thickness_um': 2.0, 'composition': {'lipid': 0.85, 'protein': 0.15}},
    {'name': 'exocuticle_sclerotised', 'thickness_um': 60.0, 'composition': {'chitin': 0.403, 'protein': 0.33, 'water': 0.18, 'melanin': 0.075, 'ommochrome': 0.012}},
    {'name': 'endocuticle', 'thickness_um': 100.0, 'composition': {'water': 0.42, 'chitin': 0.33, 'protein': 0.25}},
    # Subelytral Stagnant Air Cavity (0.162 to 0.462 mm)
    {'name': 'subelytral_air_gap', 'thickness_um': 300.0, 'composition': {'air': 1.0}},
    # Visceral Organs (0.462 to 2.913 mm)
    {'name': 'epidermis', 'thickness_um': 20.0, 'composition': {'water': 0.70, 'protein': 0.22, 'lipid': 0.08}},
    {'name': 'fat_body', 'thickness_um': 300.0, 'composition': {'lipid': 0.50, 'water': 0.38, 'protein': 0.12}},
    {'name': 'hemolymph', 'thickness_um': 250.0, 'composition': {'water': 0.93, 'protein': 0.07}},
    {'name': 'muscle', 'thickness_um': 1581.0, 'composition': {'water': 0.76, 'protein': 0.21, 'lipid': 0.03}},
    {'name': 'fat_body', 'thickness_um': 300.0, 'composition': {'lipid': 0.50, 'water': 0.38, 'protein': 0.12}},
    # Ventral Protective Shield (2.913 to 3.000 mm)
    {'name': 'epidermis', 'thickness_um': 20.0, 'composition': {'water': 0.70, 'protein': 0.22, 'lipid': 0.08}},
    {'name': 'endocuticle', 'thickness_um': 40.0, 'composition': {'water': 0.42, 'chitin': 0.33, 'protein': 0.25}},
    {'name': 'exocuticle_sclerotised', 'thickness_um': 25.0, 'composition': {'chitin': 0.43, 'protein': 0.352, 'water': 0.18, 'melanin': 0.03, 'ommochrome': 0.008}},
    {'name': 'epicuticle_wax', 'thickness_um': 2.0, 'composition': {'lipid': 0.85, 'protein': 0.15}}
]


CHROMOPHORE_HEAT_CAPACITIES = {
    'water': 0.004184,
    'chitin': 0.0021,
    'protein': 0.0022,
    'lipid': 0.0019,
    'melanin': 0.0024,
    'ommochrome': 0.0022,
    'plant_pigment': 0.0035,
    'hemocyanin': 0.0038,
    'aragonite': 0.00216,
    'conchiolin': 0.0022,
    'mucin': 0.0039,
    'hemoglobin': 0.0036,
    'air': 0.0000012
}

HUMAN_READABLE_LAYER_NAMES = {
    'epicuticle_wax': 'Epicuticular Wax Film',
    'exocuticle_sclerotised': 'Exocuticle (Sclerotised Armour)',
    'endocuticle': 'Endocuticle (Hydrated Chitin Matrix)',
    'subelytral_air_gap': 'Subelytral Air Cavity (Optothermal Barrier)',
    'epidermis': 'Epidermal Epithelium',
    'fat_body': 'Visceral Fat Body',
    'hemolymph': 'Hemolymph Vascular Sinus',
    'muscle': 'Locomotor Muscle Mass',
    'mucus': 'Mucus Mantle Film',
    'integument': 'Molluscan Integument',
    'dermal_connective': 'Dermal Connective Tissue',
    'viscera': 'Visceral Parenchyma & Organs',
    'periostracum': 'Periostracum Protein Skin',
    'shell_CaCO3': 'Aragonite Shell Matrix (CaCO3)',
    'mantle_cavity_air': 'Retracted Mantle Cavity Air Gap',
    'overlying_tissue': 'Overlying Dermal Tissue',
    'blood_vessel': 'Dorsal Vessel (Aqueous Hemolymph Sinus)',
    'dorsal_vessel': 'Dorsal Vessel (Aqueous Hemolymph Sinus)',
    'ventral_vessel': 'Ventral Hemolymph Sinus',
    'visceral_parenchyma': 'Visceral Parenchyma',
    'deep_ventral_viscera': 'Deep Ventral Viscera & Muscle',
    'deep_muscle': 'Deep Locomotor Muscle',
    'deep_ventral_tissues': 'Deep Ventral Viscera & Muscle',
    'ventral_nerve_cord': 'Ventral Nerve Cord & Parenchyma',
    'gut_wall_parenchyma': 'Midgut Wall & Parenchyma',
    'deep_tissue': 'Deep Subdermal Tissue'
}

def eval_single_layer(ld, wl):
    name = ld.get('name', 'custom_layer')
    thick_um = float(ld.get('thickness_um', 100.0))
    raw_comp = ld.get('composition', {'water': 0.70, 'protein': 0.30})
    
    COMPONENT_ALIASES = {
        'matrix': 'chitin',
        'chlorophyll': 'plant_pigment',
        'conchiolin_matrix': 'conchiolin',
        'calcite': 'aragonite',
        'blood': 'protein',
        'oxyhemoglobin': 'protein'
    }
    remapped_comp = {}
    for k, v in raw_comp.items():
        canonical_k = COMPONENT_ALIASES.get(k, k)
        remapped_comp[canonical_k] = remapped_comp.get(canonical_k, 0.0) + float(v)

    valid_comps = {}
    for k, v in remapped_comp.items():
        if k in CHROMOPHORE_HEAT_CAPACITIES:
            valid_comps[k] = max(0.0, float(v))
    total_frac = sum(valid_comps.values())
    if total_frac > 0:
        norm_comp = {k: v / total_frac for k, v in valid_comps.items()}
    else:
        norm_comp = {'water': 1.0}
        
    a_prime = float(ld.get('a_prime', 40.0))
    f_ray = float(ld.get('f_rayleigh', 0.20))
    b_mie = float(ld.get('b_mie', 1.1))
    g500 = float(ld.get('g500', 0.90))
    
    is_air = norm_comp.get('air', 0.0) >= 0.99 or name == 'subelytral_air_gap' or 'air' in name
    if is_air:
        mu_a = 0.0
        mu_s = 0.0
        mu_s_prime = 0.0
        g = 0.0
        n = 1.00029
    elif HAS_TISSUE_OPTICS:
        try:
            L = TissueLayer(
                name=name,
                thickness_um=thick_um,
                composition=norm_comp,
                a_prime=a_prime,
                f_rayleigh=f_ray,
                b_mie=b_mie,
                g500=min(0.99, g500)
            )
            mu_a = float(L.mua(wl)[0]) / 10.0        # cm^-1 -> mm^-1
            mu_s_prime = float(L.musp(wl)[0]) / 10.0   # mm^-1
            mu_s = float(L.mus(wl)[0]) / 10.0          # mm^-1
            g = float(L.g(wl)[0])
            n = float(L.n(wl)[0])
        except Exception:
            wl_val = float(wl)
            base_mua = 0.1812 * (wl_val / 450.0)**(-3.0)
            m_frac = norm_comp.get('melanin', 0.0)
            w_frac = norm_comp.get('water', 0.0)
            mu_a = base_mua + m_frac * 51.5 * (wl_val / 500.0)**(-3.48) + w_frac * 0.00092
            mu_s_prime = max(1e-4, a_prime * 0.1 * (f_ray * (wl_val / 500.0)**(-4.0) + (1.0 - f_ray) * (wl_val / 500.0)**(-b_mie)))
            g = min(0.99, g500 * ((500.0 / wl_val)**0.20 if wl_val > 500.0 else (1.0 + 0.05 * (1.0 - wl_val / 500.0))))
            mu_s = mu_s_prime / max(1.0 - g, 0.01)
            n = 1.33 * w_frac + 1.55 * (1.0 - w_frac)
    else:
        wl_val = float(wl)
        base_mua = 0.1812 * (wl_val / 450.0)**(-3.0)
        m_frac = norm_comp.get('melanin', 0.0)
        w_frac = norm_comp.get('water', 0.0)
        mu_a = base_mua + m_frac * 51.5 * (wl_val / 500.0)**(-3.48) + w_frac * 0.00092
        mu_s_prime = max(1e-4, a_prime * 0.1 * (f_ray * (wl_val / 500.0)**(-4.0) + (1.0 - f_ray) * (wl_val / 500.0)**(-b_mie)))
        g = min(0.99, g500)
        mu_s = mu_s_prime / max(1.0 - g, 0.01)
        n = 1.33 * w_frac + 1.55 * (1.0 - w_frac)

    rho_cp = sum(frac * CHROMOPHORE_HEAT_CAPACITIES.get(comp, 0.003) for comp, frac in norm_comp.items())
    
    if mu_a > 1e-5 and mu_s_prime > 1e-5:
        delta_mm = 1.0 / math.sqrt(3.0 * mu_a * (mu_a + mu_s_prime))
    elif mu_a > 1e-5:
        delta_mm = 1.0 / mu_a
    else:
        delta_mm = 999.0

    label = ld.get('label') or HUMAN_READABLE_LAYER_NAMES.get(name, name.replace('_', ' ').title())

    return {
        'name': name,
        'label': label,
        'thickness_um': thick_um,
        'thickness_mm': round(thick_um / 1000.0, 4),
        'composition': norm_comp,
        'a_prime': a_prime,
        'f_rayleigh': f_ray,
        'b_mie': b_mie,
        'g500': g500,
        'mu_a': round(mu_a, 4),
        'mu_s': round(mu_s, 3),
        'mu_s_prime': round(mu_s_prime, 3),
        'g': round(g, 3),
        'n': round(n, 3),
        'rho_cp': round(rho_cp, 6),
        'penetration_depth_mm': round(delta_mm, 3),
        'is_air': norm_comp.get('air', 0) > 0.90
    }
