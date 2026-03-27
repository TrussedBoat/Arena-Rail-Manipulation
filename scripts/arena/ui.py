def get_target_mode():
    """Helper function to ask the user which environment to target."""
    print("\n\n" + " TARGET ENVIRONMENT ".center(40, "-"))
    print("  [1] Simulation Mode (/sim/...)")
    print("  [2] Real Life Mode  (no /sim/)")
    while True:
        t_choice = input("\nSelect target (1 or 2): ").strip()
        if t_choice == '1':
            return 'sim'
        elif t_choice == '2':
            return 'real'
        else:
            print("[!] Invalid. Enter 1 or 2.")


def print_ui_header():
    print("\n" + "═"*70)
    print(" 🤖 ARENA CONTROLLER [DUAL MODE] 🤖 ".center(68, "═"))
    print("═"*70)
    print("\n ⚠️  OPERATIONAL GUIDELINES:")
    print(" ┌──────────────────────────────────────────────────────────────────┐")
    print(" │ 🧊 WHITE GOODS : Elevate adjacent tables before opening doors!   │")
    print(" │ 🛤️  ROBOT RAILS : Absolute coordinates. (+) moves toward wall.    │")
    print(" │ 🪑 TABLES (0-7): Wall Side: T0/T1 | Sitting Area Side: T6/T7     │")
    print(" │                  Odd: Door side   | Even: Window side            │")
    print(" └──────────────────────────────────────────────────────────────────┘\n")


def print_queue(queue):
    print("\n" + "📋 CURRENT SEQUENCE QUEUE ".ljust(70, "-"))
    if not queue:
        print("   (Queue is empty)")
    else:
        for i, cmd in enumerate(queue, 1):
            if cmd['type'] == 'device':
                t_str = f"[{cmd['target'].upper()}]"
                print(f"   {i}. {t_str.ljust(6)} Set [ {cmd['device'].upper()} ] -> {cmd['value']} {cmd['unit']}")
            elif cmd['type'] == 'wait':
                print(f"   {i}.        ⏱️  WAIT for {cmd['value']} seconds")
            elif cmd['type'] == 'tv_stream':
                t_str = f"[{cmd['target'].upper()}]"
                print(f"   {i}. {t_str.ljust(6)} 📺 SET TV STREAM -> {cmd['value']}")
    print("-" * 70 + "\n")
