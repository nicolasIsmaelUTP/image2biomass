import os


def get_data_path():
    """
    Determina y retorna la ruta base de los datos.
    
    Primero intenta usar la ruta local, si no existe,
    asume que se está ejecutando en Google Colab y usa la ruta de Drive.
    
    Returns:
    --------
    str
        Ruta base donde están los datos del proyecto
    """
    # Ruta local (para ejecución en entorno local)
    local_path = "../data/"
    
    # Ruta de Google Drive (para ejecución en Colab)
    colab_path = "/content/drive/MyDrive/rulookingfordata/data/"
    
    # Verificar si existe la ruta local
    if os.path.exists(local_path):
        return local_path
    
    # Si no existe local, retornar ruta de Colab
    return colab_path
