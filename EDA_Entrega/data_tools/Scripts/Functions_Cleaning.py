import pandas as pd
import numpy as np

# Diccionario de mapeo: define cómo reemplazar los valores en cada columna
mapping_dict = {
    'House': {
        'Gryffindor': 'Gryffindor',
        'Ravenclaw': 'Ravenclaw',
        'Slytherin': 'Slytherin',
        'Hufflepuff': 'Hufflepuff',
        'Beauxbatons Academy of Magic': 'Beauxbatons',
        'Durmstrang Institute': 'Durmstrang'
    },
    'Hair colour': {
        'Black': 'Black',
        'Red': 'Red',
        'Brown': 'Brown',
        'Blond': 'Blond',
        'Silver| formerly auburn': 'Silver',
        'Auburn': 'Auburn',
        'Grey': 'Grey',
        'White': 'White',
        'Light brown flecked with grey': 'Brown',
        'Colourless and balding': 'Bald',
        'Dirty-blonde': 'Blond',
        'Dark': 'Brown',  # Agrupamos "Dark" en "Brown"
        'Jet-black': 'Black',
        'Mousy': 'Brown',
        # Más agrupaciones según sea necesario
    },
    'Eye colour': {
        'Green': 'Green',
        'Blue': 'Blue',
        'Brown': 'Brown',
        'Black': 'Black',
        'Hazel': 'Hazel',
        'Grey': 'Grey',
        'Bright green': 'Green',
        'Bright brown': 'Brown',
        'Dark': 'Brown',  # Agrupamos "Dark" en "Brown"
        'Silvery': 'Grey',
        'Yellow': 'Yellow',
        'Scarlet': 'Red',
        'Grey/Blue[': 'Grey',  # Agrupación adicional
        # Más agrupaciones según sea necesario
    },
    'Species': {
        'Human': 'Human',
        'Half-Human/Half-Giant': 'Half-Human',
        'Werewolf': 'Werewolf',
        'Ghost': 'Ghost',
        'Centaur': 'Centaur',
        'Human (Metamorphmagus)': 'Human',
        'Human(goblin ancestry)': 'Human',
        # Más agrupaciones según sea necesario
    },
    # Agrega más columnas y mapeos aquí si es necesario
}

# Función para reemplazar valores en base al mapeo
def map_values(value, column_name):
    """
    Reemplaza un valor en función del mapeo definido para una columna.
    Si el valor no está en el mapeo, se pone Unknown (queda NaN si corresponde).
    Se usa para las columnas de "House", "Hair colour", "Eye colour", "Species".
    
    Args:
        value: Valor de la celda.
        column_name: Nombre de la columna correspondiente.
    
    Returns:
        El valor mapeado o "Unknown" si no se encuentra en el mapeo.
    """
    if pd.isna(value):
        return value  # Deja los NaN como están
    
    return mapping_dict[column_name].get(value, "Unknown")  # Mapea o pone "Unknown"

# Función para limpiar todo el dataframe
def map_dataframe(df):
    """
    Mapea los valores del dataframe en base al diccionario `mapping_dict`.
    
    Args:
        df (pd.DataFrame): Dataframe a limpiar.
    
    Returns:
        pd.DataFrame: Dataframe con valores mapeados.
    """
    for column in mapping_dict.keys():
        if column in df.columns:
            df[column] = df[column].apply(lambda x: map_values(x, column))

    return df



# Diccionarios para agrupar valores
loyalty_mapping = {
    'Hogwarts': ['Dumbledore\'s Army', 'Hogwarts School of Witchcraft and Wizardry'],
    'Order of the Phoenix': ['Order of the Phoenix', 'Original Order of the Phoenix', 'Albus Dumbledore'],
    'Death Eaters': ['Death Eaters', 'Lord Voldemort'],
    'Ministry of Magic': ['Ministry of Magic'],
    'Others': ['Gringotts Wizarding Bank', 'Gellert Grindelwald\'s Acolytes'],

    #'Others': [None]  # Usamos None para cubrir NaN
}


# Función para agrupar valores
def categorize_loyalty(values, mapping):
    """
    Agrupa valores en listas según un diccionario de mapeo.
    Se usa para la columna de "Loyalty".
    
    Args:
        values (list or str): Valores originales de la celda.
        mapping (dict): Diccionario de mapeo.
    
    Returns:
        list: Lista de categorías.
    """
    if pd.isna(values):  # Reemplaza los NaN por ["Unknown"]
        return ["Unknown"]
    
    # Convertir valores individuales en listas para facilitar el procesamiento
    if isinstance(values, str):
        values = [val.strip() for val in values.split('|')]

    # Crear una lista de categorías correspondientes
    categories = []
    for val in values:
        categorized = False
        for key, group in mapping.items():
            if val in group:
                categories.append(key)
                categorized = True
                break
        if not categorized:
            categories.append('Others')  # Si no coincide, lo marcamos como 'Others'
    return list(set(categories))  # Devuelve categorías únicas



def limpiar_agrupacion_blood_status(serie):
    """
    Limpia y agrupa los valores de una columna 'Blood status' en las categorías especificadas.
    
    Parámetros:
        serie (pd.Series): Columna que contiene los valores originales.
    
    Retorna:
        pd.Series: Columna con los valores agrupados.
    """
    def clasificar(status):
        """
        Clasifica el estado de sangre basado en el texto proporcionado.

        Parámetros:
            status (str or NaN): El estado de sangre como cadena de texto. Puede contener
            términos como "pure-blood", "half-blood", "muggle", etc. También puede ser NaN.

        Retorna:
            str: La categoría de estado de sangre, que puede ser "Pure-blood", "Half-blood",
            "Muggle", o "Others". Retorna NaN si el estado es NaN.
        """
        if pd.isna(status):
            return np.nan
        status = status.lower()  # Normaliza para evitar problemas de mayúsculas/minúsculas
        if "pure-blood" in status and "half-blood" not in status:
            return "Pure-blood"
        elif "half-blood" in status and "pure-blood" not in status  and "muggle-born" not in status:
            return "Half-blood"
        elif "muggle" in status:
            return "Muggle"
        else:
            return "Others"
    
    return serie.apply(clasificar)
