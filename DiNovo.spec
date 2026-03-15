# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['DiNovo.py'],
             pathex=['C:\\DiNovo_v1.5.0\\sourceCode\\'],
             binaries=[('C:\\DiNovo_v1.5.0\\sourceCode\\lib_lightgbm.dll', '.')],
             datas=[],
             hiddenimports=['lightgbm', 'sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='DiNovo',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
