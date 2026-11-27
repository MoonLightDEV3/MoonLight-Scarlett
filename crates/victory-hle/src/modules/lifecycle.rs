//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first constrains it (its
//! constrained-change routines run), then suspends it (its suspend routines
//! run, and it has a second to save) and then ends the process. Closing the
//! title's window does the same here; a new launch then starts `victory`
//! again, which waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTOPRIMARY, MONITOR_DEFAULTTONEAREST,
    MONITORINFO, MonitorFromPoint, MonitorFromWindow,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    VK_ESCAPE, VK_F2, VK_F8, VK_F11, VK_MENU, VK_RETURN,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HMENU};
use windows_sys::core::{PCSTR, PCWSTR, w};

use crate::{HleModule, hle_export};

/// A suspend routine (`quiesced`) or a constrained-change routine
/// (`constrained`): both take a flag and the title's context.
type Routine = extern "system" fn(flag: u8, context: *mut c_void);

/// Registration id, routine and context.
type Registration = (usize, Routine, usize);

/// Each suspend routine.
static ROUTINES: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

/// Each constrained-change routine.
static CONSTRAINED: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

fn routines() -> MutexGuard<'static, Vec<Registration>> {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner())
}

fn constrained() -> MutexGuard<'static, Vec<Registration>> {
    CONSTRAINED.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long the console lets suspend routines run before it ends the title.
const SUSPEND_DEADLINE: Duration = Duration::from_secs(1);

const ERROR_INVALID_PARAMETER: u32 = 87;

/// Adds `routine` to `list` and writes its registration id.
fn register(
    mut list: MutexGuard<'static, Vec<Registration>>,
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    let Some(routine) = routine else {
        return ERROR_INVALID_PARAMETER;
    };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    list.push((id, routine, context as usize));
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

pub extern "system" fn RegisterAppStateChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(routines(), routine, context, registration)
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    routines().retain(|&(id, ..)| id != registration);
}

/// The host's own version never calls the routine (a desktop app is never
/// constrained), so a title waiting for it would never suspend.
pub extern "system" fn RegisterAppConstrainedChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(constrained(), routine, context, registration)
}

pub extern "system" fn UnregisterAppConstrainedChangeNotification(registration: usize) {
    constrained().retain(|&(id, ..)| id != registration);
}

/// The window procedure a title window had before ours.
const TITLE_PROC: PCWSTR = w!("victory.wndproc");

pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: PCWSTR,
    name: PCWSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let (ex_style, style, x, y, width, height) =
        prepare_create(parent, ex_style, style, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (size and fullscreen style prepared).
    let hwnd = unsafe {
        wm::CreateWindowExW(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    hook(hwnd);
    hwnd
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: PCSTR,
    name: PCSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let (ex_style, style, x, y, width, height) =
        prepare_create(parent, ex_style, style, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (size and fullscreen style prepared).
    let hwnd = unsafe {
        wm::CreateWindowExA(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    hook(hwnd);
    hwnd
}

const FULLSCREEN_STRIP_STYLE: u32 =
    wm::WS_CAPTION | wm::WS_THICKFRAME | wm::WS_MINIMIZEBOX | wm::WS_MAXIMIZEBOX | wm::WS_SYSMENU;
const FULLSCREEN_STRIP_EX_STYLE: u32 =
    wm::WS_EX_DLGMODALFRAME | wm::WS_EX_WINDOWEDGE | wm::WS_EX_CLIENTEDGE | wm::WS_EX_STATICEDGE;

struct WindowMode {
    fullscreen: bool,
    windowed_style: u32,
    windowed_ex_style: u32,
    windowed_rect: RECT,
}

static WINDOW_MODE: Mutex<WindowMode> = Mutex::new(WindowMode {
    fullscreen: false,
    windowed_style: 0,
    windowed_ex_style: 0,
    windowed_rect: RECT {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0,
    },
});

fn start_fullscreen() -> bool {
    static FULLSCREEN: OnceLock<bool> = OnceLock::new();
    *FULLSCREEN.get_or_init(|| {
        std::env::var_os("VICTORY_FULLSCREEN").is_none_or(|v| v != "0")
            && std::env::var_os("VICTORY_WINDOW_MAX").is_none()
    })
}

fn monitor_rect_for_window(hwnd: HWND) -> Option<RECT> {
    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: valid monitor handle query and sized MONITORINFO buffer.
    let ok = unsafe {
        let mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        GetMonitorInfoW(mon, &mut info)
    };
    (ok != 0).then_some(info.rcMonitor)
}

fn primary_monitor_rect(x: i32, y: i32) -> Option<RECT> {
    let mut info = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: valid monitor handle query and sized MONITORINFO buffer.
    let ok = unsafe {
        let mon = MonitorFromPoint(POINT { x, y }, MONITOR_DEFAULTTOPRIMARY);
        GetMonitorInfoW(mon, &mut info)
    };
    (ok != 0).then_some(info.rcMonitor)
}

fn prepare_create(
    parent: HWND,
    ex_style: u32,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> (u32, u32, i32, i32, i32, i32) {
    let (wx, wy, ww, wh) = fit(parent, x, y, width, height);
    if !parent.is_null() || width < 320 || height < 200 {
        return (ex_style, style, wx, wy, ww, wh);
    }
    let fallback_style = if style & wm::WS_CAPTION != 0 {
        style
    } else {
        (style & !wm::WS_POPUP) | wm::WS_OVERLAPPEDWINDOW
    };
    let mut mode = WINDOW_MODE.lock().unwrap_or_else(|e| e.into_inner());
    mode.windowed_style = fallback_style;
    mode.windowed_ex_style = ex_style;
    mode.windowed_rect = RECT {
        left: wx,
        top: wy,
        right: wx + ww,
        bottom: wy + wh,
    };
    if start_fullscreen()
        && let Some(rc) = primary_monitor_rect(wx, wy)
    {
        mode.fullscreen = true;
        let fs_style = (style & !FULLSCREEN_STRIP_STYLE) | wm::WS_POPUP;
        let fs_ex_style = ex_style & !FULLSCREEN_STRIP_EX_STYLE;
        return (
            fs_ex_style,
            fs_style,
            rc.left,
            rc.top,
            rc.right - rc.left,
            rc.bottom - rc.top,
        );
    }
    mode.fullscreen = false;
    (ex_style, style, wx, wy, ww, wh)
}

fn toggle_fullscreen(hwnd: HWND) {
    if !is_top_level(hwnd) {
        return;
    }
    enum Action {
        Enter { style: u32, ex_style: u32, rect: RECT },
        Leave { style: u32, ex_style: u32, rect: RECT },
    }
    let action = {
        let mut mode = WINDOW_MODE.lock().unwrap_or_else(|e| e.into_inner());
        if !mode.fullscreen {
            // SAFETY: reads live window style/bounds on the window thread.
            let style = unsafe { wm::GetWindowLongPtrW(hwnd, wm::GWL_STYLE) } as u32;
            // SAFETY: as above.
            let ex_style = unsafe { wm::GetWindowLongPtrW(hwnd, wm::GWL_EXSTYLE) } as u32;
            let mut rect = RECT::default();
            // SAFETY: `rect` receives current outer window bounds.
            if unsafe { wm::GetWindowRect(hwnd, &mut rect) } != 0 {
                mode.windowed_rect = rect;
            }
            mode.windowed_style = style;
            mode.windowed_ex_style = ex_style;
            let Some(rc) = monitor_rect_for_window(hwnd) else {
                return;
            };
            mode.fullscreen = true;
            Action::Enter {
                style: (style & !FULLSCREEN_STRIP_STYLE) | wm::WS_POPUP | wm::WS_VISIBLE,
                ex_style: ex_style & !FULLSCREEN_STRIP_EX_STYLE,
                rect: rc,
            }
        } else {
            mode.fullscreen = false;
            let style = if mode.windowed_style != 0 {
                mode.windowed_style | wm::WS_VISIBLE
            } else {
                wm::WS_OVERLAPPEDWINDOW | wm::WS_VISIBLE
            };
            let mut rect = mode.windowed_rect;
            if rect.right <= rect.left || rect.bottom <= rect.top {
                let (x, y, w, h) = fit(std::ptr::null_mut(), 0, 0, 1280, 720);
                rect = RECT {
                    left: x,
                    top: y,
                    right: x + w,
                    bottom: y + h,
                };
            }
            Action::Leave {
                style,
                ex_style: mode.windowed_ex_style,
                rect,
            }
        }
    };
    match action {
        Action::Enter { style, ex_style, rect } => {
            // SAFETY: updates top-level window frame and covers the monitor in borderless mode.
            unsafe {
                wm::SetWindowLongPtrW(hwnd, wm::GWL_STYLE, style as isize);
                wm::SetWindowLongPtrW(hwnd, wm::GWL_EXSTYLE, ex_style as isize);
                wm::SetWindowPos(
                    hwnd,
                    wm::HWND_TOP,
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    wm::SWP_NOOWNERZORDER | wm::SWP_FRAMECHANGED | wm::SWP_SHOWWINDOW,
                );
            }
        }
        Action::Leave { style, ex_style, rect } => {
            // SAFETY: restores the saved windowed frame and bounds.
            unsafe {
                wm::SetWindowLongPtrW(hwnd, wm::GWL_STYLE, style as isize);
                wm::SetWindowLongPtrW(hwnd, wm::GWL_EXSTYLE, ex_style as isize);
                wm::SetWindowPos(
                    hwnd,
                    wm::HWND_NOTOPMOST,
                    rect.left,
                    rect.top,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    wm::SWP_NOOWNERZORDER | wm::SWP_FRAMECHANGED | wm::SWP_SHOWWINDOW,
                );
            }
        }
    }
    if mouse_captured() {
        clip_to_client(hwnd);
    }
}

/// A console title sizes its window to its render target (
/// the title: 3840x2160 on a Series X), which can exceed the host's screen.
/// A top-level window larger than the work area is shrunk to fit, keeping
/// its proportions, and centred; what it shows is scaled to it (the swap
/// chain stretches). `VICTORY_WINDOW_MAX=WIDTHxHEIGHT` applies a smaller
/// limit without changing the title's render targets.
fn fit(parent: HWND, x: i32, y: i32, width: i32, height: i32) -> (i32, i32, i32, i32) {
    let mut area = RECT::default();
    // SAFETY: `area` receives the primary monitor's work area.
    let known =
        unsafe { wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) };
    let limit = window_limit();
    let fitted = fit_with_limit(&area, width, height, limit);
    if !parent.is_null() || known == 0 || fitted == (width, height) {
        return (x, y, width, height);
    }
    tracing::info!(width, height, to = ?fitted, ?limit, "title window fitted to the screen");
    let (w, h) = fitted;
    let left = area.left + (area.right - area.left - w) / 2;
    let top = area.top + (area.bottom - area.top - h) / 2;
    (left, top, w, h)
}

fn parse_window_limit(value: &str) -> Option<(i32, i32)> {
    let (width, height) = value.trim().split_once(['x', 'X'])?;
    let (width, height) = (width.trim().parse().ok()?, height.trim().parse().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn window_limit() -> Option<(i32, i32)> {
    static LIMIT: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let value = std::env::var_os("VICTORY_WINDOW_MAX")?;
        let parsed = value.to_str().and_then(parse_window_limit);
        if parsed.is_none() {
            tracing::warn!(
                ?value,
                "ignoring invalid VICTORY_WINDOW_MAX; expected positive WIDTHxHEIGHT"
            );
        }
        parsed
    })
}

fn fit_with_limit(
    area: &RECT,
    width: i32,
    height: i32,
    limit: Option<(i32, i32)>,
) -> (i32, i32) {
    let Some((max_width, max_height)) = limit else {
        return fit_into(area, width, height);
    };
    let bounded = RECT {
        left: 0,
        top: 0,
        right: (area.right - area.left).min(max_width),
        bottom: (area.bottom - area.top).min(max_height),
    };
    fit_into(&bounded, width, height)
}

/// `width` x `height` scaled down to fit `area`, or unchanged if it fits
/// (or is `CW_USEDEFAULT`).
fn fit_into(area: &RECT, width: i32, height: i32) -> (i32, i32) {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if width <= 0 || height <= 0 || aw <= 0 || ah <= 0 || (width <= aw && height <= ah) {
        return (width, height);
    }
    let scale = f64::min(
        f64::from(aw) / f64::from(width),
        f64::from(ah) / f64::from(height),
    );
    (
        (f64::from(width) * scale) as i32,
        (f64::from(height) * scale) as i32,
    )
}

/// Whether client-area mouse look is captured and the host cursor should stay hidden.
static CURSOR_HIDDEN: AtomicBool = AtomicBool::new(false);
static MOUSE_CAPTURED: AtomicBool = AtomicBool::new(false);
static WAIT_LMB_UP: AtomicBool = AtomicBool::new(false);
static CAPTURED_HWND: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn set_cursor_hidden(hidden: bool) {
    CURSOR_HIDDEN.store(hidden, Ordering::Relaxed);
}

pub(crate) fn mouse_look_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_MOUSE_LOOK").is_none_or(|v| v != "0"))
}

pub(crate) fn mouse_captured() -> bool {
    MOUSE_CAPTURED.load(Ordering::Relaxed)
}

pub(crate) fn mouse_lmb_suppressed() -> bool {
    WAIT_LMB_UP.load(Ordering::Relaxed)
}

pub(crate) fn captured_hwnd() -> HWND {
    CAPTURED_HWND.load(Ordering::Relaxed) as HWND
}

fn arrow_cursor() -> wm::HCURSOR {
    static ARROW: OnceLock<usize> = OnceLock::new();
    // SAFETY: standard shared system arrow cursor handle (never destroyed).
    let raw = *ARROW.get_or_init(|| unsafe {
        wm::LoadCursorW(std::ptr::null_mut(), wm::IDC_ARROW) as usize
    });
    raw as wm::HCURSOR
}

fn is_top_level(hwnd: HWND) -> bool {
    // SAFETY: standard Win32 parent query on a live window handle.
    unsafe { wm::GetParent(hwnd) }.is_null()
}

fn clip_to_client(hwnd: HWND) {
    let mut rect = RECT::default();
    // SAFETY: `rect` receives client bounds on the window thread.
    if unsafe { wm::GetClientRect(hwnd, &mut rect) } == 0 {
        return;
    }
    let mut tl = POINT {
        x: rect.left,
        y: rect.top,
    };
    let mut br = POINT {
        x: rect.right,
        y: rect.bottom,
    };
    // SAFETY: client points converted to screen coordinates for `ClipCursor`.
    unsafe {
        if ClientToScreen(hwnd, &mut tl) == 0 || ClientToScreen(hwnd, &mut br) == 0 {
            return;
        }
        if br.x > tl.x && br.y > tl.y {
            let screen = RECT {
                left: tl.x,
                top: tl.y,
                right: br.x,
                bottom: br.y,
            };
            wm::ClipCursor(&screen);
        }
    }
}

fn center_cursor(hwnd: HWND) {
    let mut rect = RECT::default();
    // SAFETY: `rect` receives client bounds on the window thread.
    if unsafe { wm::GetClientRect(hwnd, &mut rect) } == 0 {
        return;
    }
    let mut center = POINT {
        x: (rect.left + rect.right) / 2,
        y: (rect.top + rect.bottom) / 2,
    };
    // SAFETY: converts client center to screen coordinates and centers pointer immediately.
    unsafe {
        if ClientToScreen(hwnd, &mut center) != 0 {
            wm::SetCursorPos(center.x, center.y);
        }
    }
}

fn capture_mouse(hwnd: HWND, suppress_lmb: bool) {
    CAPTURED_HWND.store(hwnd as usize, Ordering::Relaxed);
    WAIT_LMB_UP.store(suppress_lmb, Ordering::Relaxed);
    MOUSE_CAPTURED.store(true, Ordering::Relaxed);
    set_cursor_hidden(true);
    // SAFETY: hides the cursor and clips pointer travel inside the game client area.
    unsafe { wm::SetCursor(std::ptr::null_mut()) };
    clip_to_client(hwnd);
    center_cursor(hwnd);
}

fn release_mouse() {
    MOUSE_CAPTURED.store(false, Ordering::Relaxed);
    WAIT_LMB_UP.store(false, Ordering::Relaxed);
    CAPTURED_HWND.store(0, Ordering::Relaxed);
    set_cursor_hidden(false);
    // SAFETY: unclips the pointer and restores the arrow cursor.
    unsafe {
        wm::ClipCursor(std::ptr::null());
        wm::SetCursor(arrow_cursor());
    }
}

/// Puts our procedure in front of the title's.
fn hook(hwnd: HWND) {
    if hwnd.is_null() {
        return;
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = wm::GetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC);
        wm::SetPropW(hwnd, TITLE_PROC, title as *mut c_void);
        wm::SetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC, window_proc as *const () as isize);
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // A user close (close button, Alt+F4, window or taskbar menu); a
    // console title never sees one, nor a WM_CLOSE from it.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_CLOSE as usize {
        release_mouse();
        end_title("window closed", Box::new(|| {}));
        return 0;
    }
    // Alt or F10 puts a window into menu mode: the system runs a modal menu
    // loop on the thread that owns the window, which is the title's. The
    // title then stops (no frames, no input) until a key ends the loop, and
    // nothing shows it, since the window has no menu bar;
    // the title aborts itself about half a minute later. A console has no window
    // menu. The mouse still opens it from the title bar.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_KEYMENU as usize {
        return 0;
    }
    if message == wm::WM_LBUTTONDOWN && mouse_look_enabled() && !mouse_captured() && is_top_level(hwnd) {
        capture_mouse(hwnd, true);
        return 0;
    }
    if message == wm::WM_LBUTTONUP {
        WAIT_LMB_UP.store(false, Ordering::Relaxed);
    }
    if message == wm::WM_KEYDOWN
        && wparam == usize::from(VK_F11)
        && (lparam & (1 << 30)) == 0
    {
        toggle_fullscreen(hwnd);
        return 0;
    }
    if message == wm::WM_SYSKEYDOWN
        && wparam == usize::from(VK_RETURN)
        && (lparam & (1 << 29)) != 0
        && (lparam & (1 << 30)) == 0
    {
        toggle_fullscreen(hwnd);
        return 0;
    }
    if message == wm::WM_KEYDOWN
        && (wparam == usize::from(VK_F2) || wparam == usize::from(VK_F8))
        && (lparam & (1 << 30)) == 0
        && mouse_look_enabled()
    {
        if mouse_captured() {
            release_mouse();
        } else if is_top_level(hwnd) {
            capture_mouse(hwnd, false);
        }
        return 0;
    }
    if message == wm::WM_KEYDOWN
        && wparam == usize::from(VK_ESCAPE)
        && (lparam & (1 << 30)) == 0
        && mouse_captured()
    {
        release_mouse();
    }
    if message == wm::WM_SYSKEYDOWN && wparam == usize::from(VK_MENU) && mouse_captured() {
        release_mouse();
    }
    if (message == wm::WM_KILLFOCUS
        || (message == wm::WM_ACTIVATE && (wparam & 0xFFFF) == wm::WA_INACTIVE as usize)
        || message == wm::WM_NCDESTROY)
        && mouse_captured()
    {
        release_mouse();
    }
    if (message == wm::WM_SIZE || message == wm::WM_MOVE) && mouse_captured() {
        clip_to_client(hwnd);
    }
    // Console titles register WNDCLASS with hCursor = NULL and ignore WM_SETCURSOR,
    // leaving Windows' launch/busy blue spinner stuck over the client area.
    // Set the clean arrow cursor (or hide it when requested) for client hit tests,
    // leaving borders/caption to default handling.
    if message == wm::WM_SETCURSOR && ((lparam as usize) & 0xFFFF) == wm::HTCLIENT as usize {
        let cursor = if CURSOR_HIDDEN.load(Ordering::Relaxed) {
            std::ptr::null_mut()
        } else {
            arrow_cursor()
        };
        // SAFETY: standard Win32 cursor update on the window's thread.
        unsafe { wm::SetCursor(cursor) };
        return 1;
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = wm::GetPropW(hwnd, TITLE_PROC);
        if message == wm::WM_NCDESTROY {
            wm::RemovePropW(hwnd, TITLE_PROC);
        }
        let title = std::mem::transmute::<*mut c_void, wm::WNDPROC>(title);
        wm::CallWindowProcW(title, hwnd, message, wparam, lparam)
    }
}

/// Suspends the title as the console does when it ends it, runs `then`
/// (once), then ends the process. Returns at once; later calls do nothing.
pub(crate) fn end_title(why: &'static str, then: Box<dyn FnOnce() + Send>) {
    release_mouse();
    static ENDING: AtomicBool = AtomicBool::new(false);
    if ENDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let constrain = constrained().clone();
    let suspend = routines().clone();
    tracing::info!(
        constrained = constrain.len(),
        routines = suspend.len(),
        "{why}: suspending the title"
    );
    // Whoever comes first (the routines or the deadline) runs `then`; the
    // other waits for it, so the process never ends halfway through.
    let then = std::sync::Arc::new(Mutex::new(Some(then)));
    let finish = move |how: &str| {
        let mut then = then.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!("{how}; ending the title");
        if let Some(then) = then.take() {
            then();
        }
        end();
    };
    let late = finish.clone();
    // Not on the calling thread: titles hand the suspend to their window
    // thread and wait for it inside the routine.
    std::thread::spawn(move || {
        constrain_then_suspend(&constrain, &suspend);
        finish("title suspended");
    });
    std::thread::spawn(move || {
        std::thread::sleep(SUSPEND_DEADLINE);
        late("title did not finish suspending in time");
    });
}

/// Runs the constrained-change routines (constrained), then the suspend
/// routines (quiesced), as the console does when it suspends a title.
/// Grand Theft Auto VI's suspend routine waits for a state that only its
/// constrained routine brings about; called alone it never returns.
fn constrain_then_suspend(constrain: &[Registration], suspend: &[Registration]) {
    for &(_, routine, context) in constrain.iter().chain(suspend) {
        routine(1, context as *mut c_void);
    }
}

/// Set in the environment of a relaunched `victory`: an inherited handle
/// of the process it replaces.
const RELAUNCHED_FROM: &str = "VICTORY_RELAUNCHED_FROM";

/// Starts `victory` again with `args` (after the title has suspended); the
/// new process waits until this one has ended.
pub(crate) fn relaunch(args: Vec<std::ffi::OsString>) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    let mut me: HANDLE = std::ptr::null_mut();
    // SAFETY: an inheritable duplicate of our own process handle.
    let duplicated = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentProcess(),
            GetCurrentProcess(),
            &mut me,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } != 0;
    let spawned = std::env::current_exe().and_then(|exe| {
        let mut command = std::process::Command::new(exe);
        command.args(&args);
        if duplicated {
            command.env(RELAUNCHED_FROM, (me as usize).to_string());
        }
        command.spawn()
    });
    match spawned {
        Ok(child) => tracing::info!(pid = child.id(), ?args, "relaunched"),
        Err(e) => tracing::warn!(%e, "relaunch failed; the title just ends"),
    }
}

/// In a relaunched `victory`: waits (30 s at most) for the process it
/// replaces to be gone, so the two never share the title's files.
pub fn wait_for_previous() {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let Some(handle) = std::env::var(RELAUNCHED_FROM)
        .ok()
        .and_then(|h| h.parse::<usize>().ok())
    else {
        return;
    };
    // SAFETY: removing our own variable before any other thread reads it.
    unsafe { std::env::remove_var(RELAUNCHED_FROM) };
    let handle = handle as HANDLE;
    // SAFETY: the inherited process handle the previous process gave us.
    unsafe {
        WaitForSingleObject(handle, 30_000);
        CloseHandle(handle);
    }
}

fn end() {
    // Compiled host pipelines kept for the next run.
    #[cfg(windows)]
    victory_gpu::pipeline::save_pipeline_cache();
    // SAFETY: ends this process, as the console ends a suspended title.
    unsafe { TerminateProcess(GetCurrentProcess(), 0) };
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "lifecycle",
        dlls: &[
            "api-ms-win-core-psm-appnotify-*",
            "ext-ms-win-rtcore-ntuser-window-*",
        ],
        exports: vec![
            hle_export!(RegisterAppStateChangeNotification),
            hle_export!(UnregisterAppStateChangeNotification),
            hle_export!(RegisterAppConstrainedChangeNotification),
            hle_export!(UnregisterAppConstrainedChangeNotification),
            hle_export!(CreateWindowExW),
            hle_export!(CreateWindowExA),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = routines().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    #[test]
    fn constrained_registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(None, std::ptr::null_mut(), &mut a),
            ERROR_INVALID_PARAMETER
        );
        UnregisterAppConstrainedChangeNotification(a);
        let left: Vec<_> = constrained().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    /// What `recording_routine` saw: (context, flag) per call, in order.
    static CALLS: Mutex<Vec<(usize, u8)>> = Mutex::new(Vec::new());

    extern "system" fn recording_routine(flag: u8, context: *mut c_void) {
        CALLS.lock().unwrap().push((context as usize, flag));
    }

    #[test]
    fn constrained_routines_run_before_suspend_routines() {
        let r: Routine = recording_routine;
        constrain_then_suspend(&[(1, r, 10), (2, r, 11)], &[(3, r, 20), (4, r, 21)]);
        assert_eq!(*CALLS.lock().unwrap(), [(10, 1), (11, 1), (20, 1), (21, 1)]);
    }

    #[test]
    fn oversized_windows_shrink_to_the_screen() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_into(&area, 3840, 2160), (1848, 1040));
        assert_eq!(fit_into(&area, 1280, 720), (1280, 720));
        assert_eq!(
            fit_into(&area, wm::CW_USEDEFAULT, 0),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_fits_4k_into_1080p() {
        let area = RECT {
            left: 100,
            top: 50,
            right: 2660,
            bottom: 1490,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((1920, 1080))), (1920, 1080));
        assert_eq!(fit_with_limit(&area, 640, 480, Some((1920, 1080))), (640, 480));
        assert_eq!(fit_with_limit(&area, 3840, 2160, None), (2560, 1440));
    }

    #[test]
    fn configured_window_limit_still_respects_the_work_area() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((2560, 1440))), (1848, 1040));
        assert_eq!(
            fit_with_limit(&area, wm::CW_USEDEFAULT, 0, Some((1920, 1080))),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_preserves_non_widescreen_proportions() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        assert_eq!(fit_with_limit(&area, 1600, 1200, Some((1920, 1080))), (1440, 1080));
    }

    #[test]
    fn window_limit_requires_two_positive_dimensions() {
        assert_eq!(parse_window_limit("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_window_limit(" 1920 X 1080 "), Some((1920, 1080)));
        for bad in [
            "",
            "1920",
            "0x1080",
            "1920x0",
            "-1x1080",
            "1920x-1",
            "1920x1080x2",
            "2147483648x1080",
        ] {
            assert_eq!(parse_window_limit(bad), None, "{bad}");
        }
    }

    #[test]
    fn hooked_windows_still_reach_the_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("before"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        let mut text = [0u16; 16];
        // SAFETY: our window; the buffer holds 16 characters.
        unsafe {
            wm::SetWindowTextW(hwnd, w!("after"));
            let n = wm::GetWindowTextW(hwnd, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "after");
            assert!(!wm::GetPropW(hwnd, TITLE_PROC).is_null());
            wm::DestroyWindow(hwnd);
        }
    }

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
        /// The `WM_SETCURSOR` hit-test codes that reached the title's procedure.
        static CURSORS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them (no menu loop, no minimising).
    extern "system" fn recording(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == wm::WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        if message == wm::WM_SETCURSOR {
            CURSORS.with(|c| c.borrow_mut().push((lparam as usize) & 0xFFFF));
            return 0;
        }
        // SAFETY: the system's default procedure, called as the system would.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("menu"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns. `recording` stands in for the
        // title's procedure and the original is put back before the window
        // is destroyed.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            let command = |c: u32, key: u8| {
                wm::SendMessageW(hwnd, wm::WM_SYSCOMMAND, c as usize, LPARAM::from(key))
            };
            command(wm::SC_KEYMENU, 0); // Alt or F10
            command(wm::SC_KEYMENU, b' '); // Alt+Space
            command(wm::SC_KEYMENU | 1, 0); // the low bits are the system's
            command(wm::SC_MINIMIZE, 0);
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::SC_MINIMIZE as usize]);
    }

    #[test]
    fn client_setcursor_uses_host_arrow_and_skips_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("cursor"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window on this thread; `recording` stands in for the title's procedure.
        let client_res = unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            CURSORS.with(|c| c.borrow_mut().clear());
            let r = wm::SendMessageW(
                hwnd,
                wm::WM_SETCURSOR,
                hwnd as usize,
                LPARAM::from(wm::HTCLIENT as isize),
            );
            wm::SendMessageW(
                hwnd,
                wm::WM_SETCURSOR,
                hwnd as usize,
                LPARAM::from(wm::HTCAPTION as isize),
            );
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
            r
        };
        assert_eq!(client_res, 1);
        let seen = CURSORS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::HTCAPTION as usize]);
    }

    #[test]
    fn mouse_capture_tracks_hwnd_and_release_resets_it() {
        release_mouse();
        assert!(!mouse_captured());
        assert!(captured_hwnd().is_null());
        let dummy = 0x1234usize as HWND;
        CAPTURED_HWND.store(dummy as usize, Ordering::Relaxed);
        MOUSE_CAPTURED.store(true, Ordering::Relaxed);
        assert!(mouse_captured());
        assert_eq!(captured_hwnd(), dummy);
        release_mouse();
        assert!(!mouse_captured());
        assert!(captured_hwnd().is_null());
    }

    #[test]
    fn f11_toggles_borderless_fullscreen_and_restores_windowed_style() {
        let null = std::ptr::null_mut();
        // SAFETY: creates a local top-level test window with static class/caption.
        let hwnd = unsafe {
            wm::CreateWindowExW(
                0,
                w!("STATIC"),
                w!("fullscreen"),
                wm::WS_OVERLAPPEDWINDOW,
                100,
                100,
                640,
                360,
                null,
                null,
                null,
                std::ptr::null(),
            )
        };
        assert!(!hwnd.is_null());
        hook(hwnd);
        {
            let mut mode = WINDOW_MODE.lock().unwrap_or_else(|e| e.into_inner());
            mode.fullscreen = false;
        }
        // SAFETY: synchronous messages sent to a window created on this thread.
        unsafe {
            wm::SendMessageW(hwnd, wm::WM_KEYDOWN, usize::from(VK_F11), 0);
            let fs_style = wm::GetWindowLongPtrW(hwnd, wm::GWL_STYLE) as u32;
            assert_ne!(fs_style & wm::WS_POPUP, 0);
            assert_eq!(fs_style & wm::WS_CAPTION, 0);

            wm::SendMessageW(hwnd, wm::WM_KEYDOWN, usize::from(VK_F11), 0);
            let restored_style = wm::GetWindowLongPtrW(hwnd, wm::GWL_STYLE) as u32;
            assert_eq!(restored_style & wm::WS_POPUP, 0);
            assert_ne!(restored_style & wm::WS_CAPTION, 0);

            wm::DestroyWindow(hwnd);
        }
    }
}
