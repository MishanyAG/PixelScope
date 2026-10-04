APP_STYLESHEET = """
QWidget {
    background-color: #12141a;
    color: #e7eaf0;
    font-family: "Segoe UI";
    font-size: 13px;
}

QMainWindow {
    background-color: #0d0f14;
}

QFrame#sidebar {
    background-color: #0d0f14;
    border-right: 1px solid #242833;
}

QLabel#brand {
    color: #f4f6fb;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#brandSubtitle,
QLabel#muted,
QLabel#metaName {
    color: #858c9d;
}

QLabel#sectionTitle {
    color: #f0f2f7;
    font-size: 17px;
    font-weight: 600;
}

QLabel#metaValue {
    color: #dfe3eb;
    font-weight: 600;
}

QPushButton {
    border: 1px solid #2b303c;
    border-radius: 8px;
    background-color: #1a1d25;
    padding: 9px 13px;
}

QPushButton:hover {
    background-color: #222631;
    border-color: #393f4d;
}

QPushButton:pressed {
    background-color: #171a21;
}

QPushButton#primaryButton {
    background-color: #e8ebf2;
    color: #111318;
    border: none;
    font-weight: 700;
    padding: 10px 16px;
}

QPushButton#primaryButton:hover {
    background-color: #ffffff;
}

QPushButton[nav="true"] {
    text-align: left;
    border: none;
    background: transparent;
    color: #9299aa;
    padding: 10px 14px;
}

QPushButton[nav="true"]:hover {
    background-color: #171a21;
    color: #e7eaf0;
}

QPushButton[nav="true"][active="true"] {
    background-color: #20242d;
    color: #ffffff;
    font-weight: 600;
}

QPushButton[nav="true"]:disabled {
    color: #555b68;
    background: transparent;
}

QFrame#topbar {
    background-color: #12141a;
    border-bottom: 1px solid #242833;
}

QFrame#inspector {
    background-color: #101218;
    border-left: 1px solid #242833;
}

QFrame#imageViewer {
    background-color: #0b0d11;
    border: 1px solid #252a35;
    border-radius: 12px;
}

QLabel#viewerHint {
    color: #747c8f;
    font-size: 14px;
}

QStatusBar {
    background-color: #0d0f14;
    color: #777f90;
    border-top: 1px solid #242833;
}

QMenuBar {
    background-color: #0d0f14;
}

QMenuBar::item:selected,
QMenu::item:selected {
    background-color: #252a35;
}

QMenu {
    background-color: #151820;
    border: 1px solid #2a2f3a;
}
"""
