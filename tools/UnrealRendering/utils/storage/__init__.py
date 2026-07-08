from .base import Storage



def get_storage_instance(storage_type, project_path, args):
    if storage_type == 'local':
        from .local import LocalStorage
        return LocalStorage(args, project_path)
    else:
        return Storage(args, project_path)