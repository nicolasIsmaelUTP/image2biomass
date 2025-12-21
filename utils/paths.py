import os


def get_data_path():
    """
    Determina y retorna la ruta base de los datos.
    
    Primero intenta usar la ruta local relativa, si no existe,
    intenta usar ruta absoluta en el directorio actual (Colab después de chdir),
    y finalmente usa la ruta de Drive.
    
    Returns:
    --------
    str
        Ruta base donde están los datos del proyecto
    """
    # Ruta local relativa (para ejecución en entorno local desde notebooks/)
    local_path = "../data/"
    
    # Ruta absoluta en el directorio actual (para Colab después de cambiar directorio)
    absolute_path = "data/"
    
    # Ruta de Google Drive (fallback para Colab)
    colab_path = "/content/drive/MyDrive/rulookingfordata/data/"
    
    # Verificar las rutas en orden de preferencia
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(absolute_path):
        return absolute_path
    else:
        # Si no existe ninguna, retornar ruta de Colab
        return colab_path
