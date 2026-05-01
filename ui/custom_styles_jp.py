import tkinter as tk
import tkinter.ttk as ttk

def setup_ttk_styles(master=None):
    # 統一フォント
    jp_font = ("Meiryo", 12)
    jp_font_bold = ("Meiryo", 12, "bold")

    # 統一余白・レイアウト方針（pack前提）
    PACK_LAYOUT = {
        'expand': True,
        'fill': 'both',
        'side': 'top',
        'padx': 10,
        'pady': 5
    }

    style = ttk.Style(master)

    # ========== TButton (flat) ==========

    style.configure("primary.TButton",
                    font=jp_font_bold,
                    # foreground="blue",
                    # background="red",
                    padding=(12, 6),
                    relief="flat"
                    )
    # style.map("primary.TButton",
    #           background=[("active", "#005A9E")],
    #           foreground=[("disabled", "#999999")]
    #           )


    style.configure("secondary.TButton",
                    font=jp_font_bold,
                    # foreground="white",
                    # background="#6C757D",
                    padding=(12, 6),
                    anchor="center",
                    relief="flat",
                    borderwidth=0)
    style.map("secondary.TButton",
              background=[('active', '#5A6268'), ('pressed', '#495057')],
              foreground=[('disabled', '#999999')])

    # ========== TCheckbutton ==========
    style.configure("custom.TCheckbutton",
                    font=jp_font,
                    padding=6,
                    anchor="w")  # 左寄せだがpackでは中央に積む想定

    # ========== TRadiobutton ==========
    style.configure("custom.TRadiobutton",
                    font=jp_font,
                    padding=6,
                    anchor="w")

    # ========== TLabel ==========
    style.configure("custom.TLabel",
                    font=jp_font,
                    anchor="w",
                    padding=4)

    # ========== TLabelframe ==========
    style.configure("custom.TLabelframe",
                    font=jp_font_bold,
                    relief="groove",
                    # background="#F0F4F8",
                    borderwidth=0)

    style.configure("custom.TLabelframe.Label",
                    font=jp_font_bold,
                    # background="#F0F4F8",
                    # foreground="#333333",
                    padding=(4, 0))

    # ========== TFrame ==========
    style.configure("custom.TFrame",
                    # background="#F8F9FA",
                    )

    # ========== TEntry ==========
    style.configure("custom.TEntry",
                    font=jp_font,
                    padding=4)

    # オプション：統一layout dictを返す（各 pack 呼び出しに使用可能）
    return PACK_LAYOUT
