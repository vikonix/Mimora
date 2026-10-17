# Installs Mimora on Windows with uv, for a user who uses a terminal.
#
# Install or upgrade:
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/vikonix/Mimora/releases/latest/download/install.ps1 | iex"
#
# Uninstall (irm | iex cannot pass arguments, so the script runs as a script block):
#   powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://github.com/vikonix/Mimora/releases/latest/download/install.ps1))) -Uninstall"
#
# Environment variables:
#   MIMORA_VERSION      install this exact version, for example 1.2.0rc1.
#                       After an exact version, "uv tool upgrade mimora" does
#                       not upgrade; run this script again without the variable.
#   MIMORA_NO_SHORTCUT  any value: do not create the Start menu shortcut.
#
# The models are not downloaded here: the first start of Mimora asks for them.
#
# Keep this file ASCII only: Windows PowerShell 5.1 reads a file without a BOM
# in the ANSI code page.

param(
    [switch]$Uninstall
)

# Must be the same value as APP_USER_MODEL_ID in mimora\window_icon.py and
# AppUserModelID in installer\mimora.iss. With different values, a pinned
# shortcut and the running window become two separate taskbar buttons.
$MimoraAppUserModelId = 'vikonix.Mimora'

$MimoraShortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Mimora.lnk'

function Write-MimoraStep([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-MimoraWarning([string]$Message) {
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

# Returns the full path of uv.exe, or $null. The full path is necessary
# because the uv installer changes PATH only for new terminal windows.
function Find-MimoraUv {
    $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) {
        return $command.Source
    }
    # The directories where the official uv installer puts uv.exe, in the
    # order that the installer itself uses.
    $candidates = @()
    if ($env:UV_INSTALL_DIR) { $candidates += $env:UV_INSTALL_DIR }
    if ($env:XDG_BIN_HOME) { $candidates += $env:XDG_BIN_HOME }
    $candidates += (Join-Path $env:USERPROFILE '.local\bin')
    foreach ($directory in $candidates) {
        $path = Join-Path $directory 'uv.exe'
        if (Test-Path -LiteralPath $path) {
            return $path
        }
    }
    return $null
}

function Install-MimoraUv {
    Write-MimoraStep 'Installing uv (the official installer from Astral)'
    # A child process, so that the variables and functions of the uv
    # installer do not mix with the ones of this script.
    & powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if ($LASTEXITCODE -ne 0) {
        throw "The uv installation failed (exit code $LASTEXITCODE). Read the messages above."
    }
    $uv = Find-MimoraUv
    if (-not $uv) {
        throw 'uv was installed, but uv.exe was not found. Open a new terminal and run this command again.'
    }
    return $uv
}

# torch and llama-server do not load without the Visual C++ runtime, and a
# clean Windows does not have it.
function Test-MimoraVcRuntime {
    # A 32-bit PowerShell sees SysWOW64 (the 32-bit runtime) under System32.
    # Sysnative is the real System32 for a 32-bit process.
    if ([Environment]::Is64BitProcess) {
        $systemDir = Join-Path $env:SystemRoot 'System32'
    } else {
        $systemDir = Join-Path $env:SystemRoot 'Sysnative'
    }
    foreach ($dll in @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $systemDir $dll))) {
            return $false
        }
    }
    return $true
}

function Install-MimoraVcRuntime {
    Write-MimoraStep 'Installing the Microsoft Visual C++ runtime (Windows asks for permission)'
    $installer = Join-Path ([IO.Path]::GetTempPath()) "mimora-vc_redist-$PID.x64.exe"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $installer
        try {
            $process = Start-Process -FilePath $installer -ArgumentList '/install', '/passive', '/norestart' -Wait -PassThru
        } catch {
            # Start-Process throws when the user refuses the UAC prompt.
            throw 'The Visual C++ runtime was not installed. Mimora cannot start without it. Run this command again and allow the installation.'
        }
        # 3010: installed, a restart is required later.
        # 1638: a newer version is already installed.
        if ($process.ExitCode -notin @(0, 3010, 1638)) {
            throw "The Visual C++ runtime installer failed (exit code $($process.ExitCode))."
        }
    } finally {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    }
}

# Makes the uv tool bin directory available as a command in new terminals.
function Add-MimoraBinToPath([string]$Uv, [string]$BinDir) {
    $normalized = $BinDir.TrimEnd('\')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @()
    if ($userPath) {
        $entries = $userPath -split ';' | ForEach-Object { $_.TrimEnd('\') }
    }
    if ($entries -notcontains $normalized) {
        Write-MimoraStep "Adding $BinDir to the user PATH"
        & $Uv tool update-shell
        if ($LASTEXITCODE -ne 0) {
            Write-MimoraWarning "Could not add $BinDir to PATH. The Start menu shortcut works without it."
        }
    }
    # The current window too, so that "mimora" works at once.
    if (($env:Path -split ';' | ForEach-Object { $_.TrimEnd('\') }) -notcontains $normalized) {
        $env:Path = "$BinDir;$env:Path"
    }
}

# Writes System.AppUserModel.ID into an existing shortcut. WScript.Shell
# cannot set this property, so the call goes to the shell property store.
function Set-MimoraShortcutAppId([string]$ShortcutPath, [string]$AppId) {
    # Add-Type fails if the type already exists, for example when the script
    # runs a second time in the same window.
    if (-not ('MimoraSetup.ShortcutAppId' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace MimoraSetup
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey
    {
        public Guid FormatId;
        public int PropertyId;
    }

    // Only the VT_LPWSTR form is used. The last field makes the structure
    // as large as the real PROPVARIANT on 64-bit Windows.
    [StructLayout(LayoutKind.Explicit)]
    public struct PropVariant
    {
        [FieldOffset(0)] public ushort VarType;
        [FieldOffset(8)] public IntPtr Value;
        [FieldOffset(16)] public IntPtr Reserved;
    }

    [ComImport]
    [Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IPropertyStore
    {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    public static class ShortcutAppId
    {
        private const int GPS_READWRITE = 2;
        private const ushort VT_LPWSTR = 31;

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void SHGetPropertyStoreFromParsingName(
            string path, IntPtr bindContext, int flags, ref Guid iid,
            [MarshalAs(UnmanagedType.Interface)] out IPropertyStore store);

        public static void Set(string shortcutPath, string appId)
        {
            Guid iid = typeof(IPropertyStore).GUID;
            IPropertyStore store;
            SHGetPropertyStoreFromParsingName(shortcutPath, IntPtr.Zero, GPS_READWRITE, ref iid, out store);
            // PKEY_AppUserModel_ID
            PropertyKey key = new PropertyKey();
            key.FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");
            key.PropertyId = 5;
            PropVariant value = new PropVariant();
            value.VarType = VT_LPWSTR;
            value.Value = Marshal.StringToCoTaskMemUni(appId);
            try
            {
                store.SetValue(ref key, ref value);
                store.Commit();
            }
            finally
            {
                Marshal.FreeCoTaskMem(value.Value);
                Marshal.ReleaseComObject(store);
            }
        }
    }
}
'@
    }
    [MimoraSetup.ShortcutAppId]::Set($ShortcutPath, $AppId)
}

function New-MimoraShortcut([string]$Uv, [string]$BinDir) {
    $target = Join-Path $BinDir 'mimora-gui.exe'
    if (-not (Test-Path -LiteralPath $target)) {
        Write-MimoraWarning "$target was not found, so no shortcut was created."
        return
    }
    Write-MimoraStep "Creating the Start menu shortcut $MimoraShortcutPath"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($MimoraShortcutPath)
    $shortcut.TargetPath = $target
    $shortcut.WorkingDirectory = $BinDir
    $shortcut.Description = 'Mimora pronunciation trainer'
    # The uv launcher exe has no icon of its own. The icon comes with the
    # package; "Lib" in a Windows virtual environment has no version number,
    # so the path stays the same after an upgrade.
    $toolDir = (& $Uv tool dir).Trim()
    $icon = Join-Path $toolDir 'mimora\Lib\site-packages\mimora\icons\mimora.ico'
    if (Test-Path -LiteralPath $icon) {
        $shortcut.IconLocation = "$icon,0"
    }
    $shortcut.Save()

    # Without the ID the shortcut still starts Mimora. Only the taskbar
    # grouping is wrong, so a failure here does not stop the installation.
    try {
        Set-MimoraShortcutAppId $MimoraShortcutPath $MimoraAppUserModelId
    } catch {
        Write-MimoraWarning "Could not set the taskbar ID of the shortcut: $($_.Exception.Message)"
        Write-MimoraWarning 'If you pin Mimora to the taskbar, the pinned button and the window can show as two buttons.'
    }
}

function Get-MimoraDataDir {
    if ($env:MIMORA_HOME) {
        return $env:MIMORA_HOME
    }
    if ($env:APPDATA) {
        return Join-Path $env:APPDATA 'Mimora'
    }
    return Join-Path $env:USERPROFILE 'AppData\Roaming\Mimora'
}

function Install-Mimora {
    $uv = Find-MimoraUv
    if ($uv) {
        Write-MimoraStep "Using uv: $uv"
    } else {
        $uv = Install-MimoraUv
    }

    if (Test-MimoraVcRuntime) {
        Write-MimoraStep 'The Visual C++ runtime is installed'
    } else {
        Install-MimoraVcRuntime
    }

    $package = 'mimora'
    if ($env:MIMORA_VERSION) {
        $package = "mimora==$($env:MIMORA_VERSION)"
    }
    Write-MimoraStep "Installing $package (Python, PyTorch and the dependencies, this can take several minutes)"
    # --upgrade: a second run of this script is an upgrade.
    # --managed-python: always the Python build of uv, never a Python that is
    #   already on the computer and can be a version that Mimora does not support.
    # --python 3.12: a uv tool environment ignores requires-python.
    # --torch-backend auto: without it an NVIDIA computer gets a CPU-only torch.
    & $uv tool install --upgrade --managed-python --python 3.12 --torch-backend auto $package
    if ($LASTEXITCODE -ne 0) {
        throw "The Mimora installation failed (exit code $LASTEXITCODE). Read the messages above. If uv reports an unknown option, update uv ('uv self update') and run this command again."
    }

    $binDir = (& $uv tool dir --bin).Trim()
    Add-MimoraBinToPath $uv $binDir

    if ($env:MIMORA_NO_SHORTCUT) {
        Write-MimoraStep 'MIMORA_NO_SHORTCUT is set: no shortcut was created'
    } else {
        New-MimoraShortcut $uv $binDir
    }

    Write-Host ''
    Write-Host 'Mimora is installed.' -ForegroundColor Green
    Write-Host '  Start it from the Start menu, or run "mimora" in a new terminal.'
    Write-Host '  At the first start, Mimora asks to download its models (several GB).'
    Write-Host '  Upgrade:   uv tool upgrade mimora   (or run this command again)'
    Write-Host '  Uninstall: see the top of this script, or run "uv tool uninstall mimora" and delete the shortcut.'
}

function Uninstall-Mimora {
    $uv = Find-MimoraUv
    if ($uv) {
        Write-MimoraStep 'Removing the Mimora uv tool'
        & $uv tool uninstall mimora
        if ($LASTEXITCODE -ne 0) {
            Write-MimoraWarning 'uv could not remove Mimora. Possibly it was not installed with uv.'
        }
    } else {
        Write-MimoraWarning 'uv was not found, so the Mimora uv tool was not removed.'
    }

    if (Test-Path -LiteralPath $MimoraShortcutPath) {
        Write-MimoraStep "Deleting $MimoraShortcutPath"
        Remove-Item -LiteralPath $MimoraShortcutPath -Force
    }

    # The data directory is not deleted: it can hold several GB of models
    # that a later installation uses again, and MIMORA_HOME can point to a
    # directory that also holds other files of the user.
    Write-Host ''
    Write-Host 'Mimora is uninstalled.' -ForegroundColor Green
    $dataDir = Get-MimoraDataDir
    if (Test-Path -LiteralPath $dataDir) {
        Write-Host "  The models, settings and logs stay in: $dataDir"
        Write-Host '  Delete this directory yourself if you do not need them.'
    }
    Write-Host '  uv itself stays installed. To remove it, see https://docs.astral.sh/uv/'
}

# All the work is in functions, and this is the only line that starts it.
# When "irm | iex" receives only part of the file, this line is missing and
# nothing runs.
& {
    # Inside a script block, so that "irm | iex" does not change the error
    # preference of the user's window.
    $ErrorActionPreference = 'Stop'
    # The progress bar of Invoke-WebRequest in Windows PowerShell 5.1 makes
    # the download of vc_redist many times slower.
    $ProgressPreference = 'SilentlyContinue'
    # Windows PowerShell 5.1 does not always enable TLS 1.2, and GitHub and
    # aka.ms refuse older protocols.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if ($Uninstall) {
        Uninstall-Mimora
    } else {
        Install-Mimora
    }
}
